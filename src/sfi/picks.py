"""Paper-tracking picks: research tracking, NOT wagering advice.

Each live run writes `ledger/paper_picks/picks_<date>.csv` (date = run date, US Eastern), one row per
forecast game. The pick is the side where `elo_tuned`'s probability most exceeds the market's implied
probability (raw implied from the American odds at issue time, so it includes the bookmaker margin).
Stake, confidence and the pick are paper quantities used to test whether these "edges" are real.

Out of sample (2020-2025) `elo_tuned` was clearly worse than the moneyline market (Brier +0.013), so most
apparent edges are expected to be model error. `sfi score` grades the paper picks: simulated profit at
the archived odds, return per unit staked with a week-block bootstrap interval, and closing-line value.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from sfi.data.schema import Game

COLUMNS = ["Pick ID", "Date", "Week", "Sport", "Away", "Home", "Bet Type", "Pick", "Line", "Odds (American)",
           "AI Win Prob", "Confidence", "Stake (units)", "AI Reasoning (summary)"]
EXTRA = ["Paper Only", "Issued At (UTC)", "Run ID", "Model", "Edge", "Market Implied P (raw)", "Game ID", "Pick Side"]
DISCLAIMER = "Paper tracking for research; not wagering advice."


def american_implied(odds: float) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def american_profit(odds: float, stake: float) -> float:
    """Profit (excluding stake) of a winning bet."""
    return stake * (odds / 100 if odds > 0 else 100 / -odds)


def make_picks(*, run: dict, p_home: dict[str, float], games: dict[str, Game], model: str = "elo_tuned",
               stake_edge: float = 0.03, high: float = 0.10, medium: float = 0.05, eastern=None) -> list[dict]:
    lines = run.get("market_lines_at_issue", {})
    rows = []
    for gid in run["game_ids"]:
        if gid not in p_home:
            continue
        g, p = games[gid], p_home[gid]
        ln = lines.get(gid, {})
        h_ml, a_ml = ln.get("home_ml"), ln.get("away_ml")
        if h_ml is not None and a_ml is not None:
            q_h, q_a = american_implied(h_ml), american_implied(a_ml)
            e_h, e_a = p - q_h, (1 - p) - q_a
            side = "home" if e_h >= e_a else "away"
            edge = e_h if side == "home" else e_a
            odds = h_ml if side == "home" else a_ml
            q = q_h if side == "home" else q_a
        else:  # no prices: model favorite, no edge, no stake
            side = "home" if p >= 0.5 else "away"
            edge = odds = q = None
        team = g.home_team if side == "home" else g.away_team
        p_pick = p if side == "home" else 1 - p
        stake = 1 if (edge is not None and edge >= stake_edge) else 0
        conf = "Low" if edge is None else "High" if edge >= high else "Medium" if edge >= medium else "Low"
        if edge is None:
            reason = (f"{DISCLAIMER} {model} P({team})={p_pick:.3f}; no moneyline available, so no edge "
                      f"and no paper stake.")
        else:
            reason = (f"{DISCLAIMER} {model} P({team})={p_pick:.3f} vs market implied {q:.3f} at {odds:+g} "
                      f"(includes bookmaker margin): edge {edge:+.3f}. {model} trailed the market out of sample "
                      f"(Brier +0.013), so most edges are expected to be model error.")
        kickoff = g.scheduled_start_utc.astimezone(eastern) if eastern else g.scheduled_start_utc
        rows.append({
            "Pick ID": f"{g.season}-{g.week:02d}-{g.away_team}-{g.home_team}-ML", "Date": kickoff.strftime("%Y-%m-%d"),
            "Week": g.week, "Sport": "NFL", "Away": g.away_team, "Home": g.home_team, "Bet Type": "Moneyline",
            "Pick": team, "Line": "", "Odds (American)": "" if odds is None else f"{odds:+g}",
            "AI Win Prob": f"{p_pick:.4f}", "Confidence": conf, "Stake (units)": stake,
            "AI Reasoning (summary)": reason, "Paper Only": "yes", "Issued At (UTC)": run["issued_at_utc"],
            "Run ID": run["run_id"], "Model": model, "Edge": "" if edge is None else f"{edge:.4f}",
            "Market Implied P (raw)": "" if q is None else f"{q:.4f}", "Game ID": gid, "Pick Side": side,
        })
    return rows


def write_picks(rows: list[dict], folder: Path, run_date: str) -> Path:
    """Appends to picks_<run_date>.csv (several runs on one day each add their rows)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"picks_{run_date}.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS + EXTRA, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(rows)
    return path


def read_all_picks(folder: Path) -> list[dict]:
    rows = []
    for path in sorted(folder.glob("picks_*.csv")):
        with open(path, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def latest_pre_kickoff(rows: list[dict], games: dict[str, Game]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        g = games.get(r["Game ID"])
        if g is None:
            continue
        issued = datetime.strptime(r["Issued At (UTC)"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=g.scheduled_start_utc.tzinfo)
        if issued >= g.scheduled_start_utc:
            continue
        if r["Pick ID"] not in out or r["Issued At (UTC)"] > out[r["Pick ID"]]["Issued At (UTC)"]:
            out[r["Pick ID"]] = r
    return out


def grade(rows: list[dict], games: dict[str, Game], closing: dict[str, dict], n_boot: int = 2000,
          seed: int = 0) -> dict:
    """Grades the latest pre-kickoff paper pick per game. Returns counts, profit, return per unit staked
    with a week-block bootstrap CI, closing-line value (raw implied probability at close minus at issue
    for the picked side; positive = the market moved toward the pick), and calibration of the pick prob."""
    from sfi.evaluation.metrics import block_bootstrap_mean_ci

    picks = latest_pre_kickoff(rows, games)
    staked, clv, probs, hits = [], [], [], []
    pending = 0
    for pid, r in picks.items():
        g = games[r["Game ID"]]
        if not g.is_final:
            pending += 1
            continue
        won = (g.home_outcome == 1.0) if r["Pick Side"] == "home" else (g.home_outcome == 0.0)
        tie = g.home_outcome == 0.5
        if not tie:
            probs.append(float(r["AI Win Prob"]))
            hits.append(float(won))
        block = f"{g.season}-W{g.week:02d}"
        if r["Odds (American)"]:
            close = closing.get(r["Game ID"], {})
            c_odds = close.get("home_ml" if r["Pick Side"] == "home" else "away_ml")
            if c_odds is not None:
                clv.append((american_implied(c_odds) - float(r["Market Implied P (raw)"]), block))
        stake = float(r["Stake (units)"])
        if stake > 0 and r["Odds (American)"]:
            profit = 0.0 if tie else (american_profit(float(r["Odds (American)"]), stake) if won else -stake)
            staked.append((profit / stake, block, profit, stake))
    out = {"picks_graded": len(probs), "pending": pending, "staked_bets": len(staked)}
    if staked:
        roi, lo, hi = block_bootstrap_mean_ci([x[0] for x in staked], [x[1] for x in staked], n_boot, seed=seed)
        out.update(units_staked=sum(x[3] for x in staked), units_profit=round(sum(x[2] for x in staked), 3),
                   roi_per_unit=roi, roi_ci95=[lo, hi])
    if clv:
        m, lo, hi = block_bootstrap_mean_ci([x[0] for x in clv], [x[1] for x in clv], n_boot, seed=seed)
        out.update(clv_mean=m, clv_ci95=[lo, hi], clv_n=len(clv))
    if probs:
        out.update(mean_pick_prob=float(np.mean(probs)), pick_hit_rate=float(np.mean(hits)))
    return out
