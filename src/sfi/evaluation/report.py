"""Score archived forecasts against final results and write a Markdown evaluation report."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sfi.data.schema import Game
from sfi.evaluation import metrics as M
from sfi.forecasts.record import ForecastRecord

ORACLE = "oracle_true_prob"


@dataclass
class ScoredSet:
    model: str
    game_ids: list[str] = field(default_factory=list)
    p: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    block: list[str] = field(default_factory=list)
    n_tie: int = 0
    n_abstain: int = 0


def score(
    records: list[ForecastRecord], games: dict[str, Game], seasons: set[int],
    benchmarks: dict[str, dict[str, float]] | None = None,
) -> dict[str, ScoredSet]:
    """Score records whose season is in `seasons`; everything else is ignored.

    `benchmarks` are reference probabilities that are not model forecasts (the synthetic oracle, the
    betting market). Each is scored on the games it covers among those the models scored.
    """
    sets: dict[str, ScoredSet] = {}
    for r in records:
        if r.season not in seasons:
            continue
        s = sets.setdefault(r.model_name, ScoredSet(r.model_name))
        g = games[r.game_id]
        if r.quality_gate.status != "publish":
            s.n_abstain += 1
        elif g.home_outcome == 0.5:
            s.n_tie += 1
        else:
            s.game_ids.append(g.game_id)
            s.p.append(r.distribution["p_home_win"])
            s.y.append(g.home_outcome)
            s.block.append(f"{g.season}-W{g.week:02d}")
    if benchmarks and sets:
        ref = max(sets.values(), key=lambda s: len(s.p))
        for name, probs in benchmarks.items():
            o = ScoredSet(name, n_tie=ref.n_tie, n_abstain=ref.n_abstain)
            for gid, y, b in zip(ref.game_ids, ref.y, ref.block):
                if gid in probs:
                    o.game_ids.append(gid)
                    o.p.append(probs[gid])
                    o.y.append(y)
                    o.block.append(b)
            sets[name] = o
    return sets


def summarize(sets: dict[str, ScoredSet], n_boot: int, seed: int = 0) -> dict[str, dict]:
    out = {}
    for name, s in sets.items():
        if not s.p:
            out[name] = {"n_scored": 0}
            continue
        b, b_lo, b_hi = M.block_bootstrap_mean_ci(M.brier_terms(s.p, s.y), s.block, n_boot, seed=seed)
        a, sl = M.calibration_intercept_slope(s.p, s.y)
        out[name] = {
            "n_scored": len(s.p), "n_tie": s.n_tie, "n_abstain": s.n_abstain,
            "brier": b, "brier_ci95": [b_lo, b_hi], "log_loss": M.log_loss(s.p, s.y),
            "calibration_intercept": None if np.isnan(a) else a,
            "calibration_slope": None if np.isnan(sl) else sl,
            "accuracy_at_0.5": float(np.mean((np.asarray(s.p) > 0.5) == (np.asarray(s.y) == 1))),
        }
    return out


def paired_brier_diff(sets: dict[str, ScoredSet], a: str, b: str, n_boot: int, seed: int = 0):
    """Brier(a) - Brier(b) on the games both scored, CI by week-block bootstrap. Negative favors a.
    Returns (mean, lo, hi, n_games)."""
    sa, sb = sets[a], sets[b]
    ib = {gid: i for i, gid in enumerate(sb.game_ids)}
    pairs = [(i, ib[gid]) for i, gid in enumerate(sa.game_ids) if gid in ib]
    if not pairs:
        raise ValueError(f"{a} and {b} share no scored games")
    ia, jb = zip(*pairs)
    pa, ya = [sa.p[i] for i in ia], [sa.y[i] for i in ia]
    pb, yb = [sb.p[j] for j in jb], [sb.y[j] for j in jb]
    if ya != yb:
        raise ValueError("outcomes differ for the same game")
    d = M.brier_terms(pa, ya) - M.brier_terms(pb, yb)
    m, lo, hi = M.block_bootstrap_mean_ci(d, [sa.block[i] for i in ia], n_boot, seed=seed)
    return m, lo, hi, len(pairs)


def section_lines(
    title: str, note: str, sets: dict[str, ScoredSet], summary: dict[str, dict],
    comparisons: list[tuple[str, str]], reliability: list[str], n_boot: int, n_bins: int,
) -> list[str]:
    lines = [f"## {title}", "", note, "",
             "| Model | Scored | Brier | Brier 95% CI | Log loss | Calib. intercept | Calib. slope | Acc @0.5 | Ties | Abstained |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in summary.items():
        if s.get("n_scored", 0) == 0:
            lines.append(f"| {name} | 0 | – | – | – | – | – | – | – | – |")
            continue
        lo, hi = s["brier_ci95"]
        a, sl = s["calibration_intercept"], s["calibration_slope"]
        a_txt, sl_txt = ("n/a", "n/a") if sl is None else (f"{a:+.3f}", f"{sl:.3f}")
        lines.append(
            f"| {name} | {s['n_scored']} | {s['brier']:.4f} | [{lo:.4f}, {hi:.4f}] | {s['log_loss']:.4f} | "
            f"{a_txt} | {sl_txt} | {s['accuracy_at_0.5']:.3f} | {s['n_tie']} | {s['n_abstain']} |")

    lines += ["", "Paired Brier differences on games both scored (week-block bootstrap; negative favors A):", "",
              "| A − B | Games | Mean diff | 95% CI | Reading |", "|---|---:|---:|---|---|"]
    for a, b in comparisons:
        if a not in sets or b not in sets or not sets[a].p or not sets[b].p:
            continue
        m, lo, hi, n = paired_brier_diff(sets, a, b, n_boot)
        reading = "A better" if hi < 0 else "B better" if lo > 0 else "no clear difference"
        lines.append(f"| {a} − {b} | {n} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {reading} |")

    for model in reliability:
        if model not in sets or not sets[model].p:
            continue
        s = sets[model]
        lines += ["", f"Reliability table: `{model}`", "",
                  "| Forecast bin | Games | Mean forecast | Observed home-win rate |", "|---|---:|---:|---:|"]
        for c in M.calibration_table(s.p, s.y, n_bins):
            lines.append(f"| {c.lo:.1f}–{c.hi:.1f} | {c.n} | {c.mean_p:.3f} | {c.observed:.3f} |")
    return lines + [""]


BENCHMARK_NOTES = {
    ORACLE: f"`{ORACLE}` uses the generator's hidden true probabilities: the best any model could do on "
            "this synthetic data, and **not available for real data**.",
    "market_moneyline": "`market_moneyline` is the betting market's implied probability (moneylines with "
                        "the bookmaker margin removed). It is a **benchmark only**, never a model input; "
                        "the time the line was recorded is undocumented (likely near kickoff), so it may "
                        "contain information our forecasts could not have had. Covered games only.",
}


def write_markdown(
    path: str | Path, *, title: str, data_kind: str, header: list[str], sections: list[list[str]],
    benchmarks: list[str] = (),
) -> None:
    lines = [f"# {title}", ""]
    if data_kind == "synthetic":
        lines += ["> **SYNTHETIC DATA — NOT REAL NFL RESULTS.** Teams, games and scores are generated",
                  "> by `sfi.data.synthetic`. These numbers test the pipeline; they say nothing about",
                  "> real-world forecasting skill.", ""]
    else:
        lines += ["> **Real NFL results, retrospective backtest.** Forecasts are replayed from historical",
                  "> data under a known-at policy, not made live before the games.", ""]
    lines += [f"- {h}" for h in header]
    notes = ["Calibration slope/intercept are n/a for near-constant forecasts (not identifiable). "
             "A slope below 1 means forecasts are too extreme; above 1, too timid. Reliability bins "
             "with few games are noisy."]
    notes += [BENCHMARK_NOTES[b] for b in benchmarks if b in BENCHMARK_NOTES]
    lines += [""] + [n + "\n" for n in notes]
    for s in sections:
        lines += s
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
