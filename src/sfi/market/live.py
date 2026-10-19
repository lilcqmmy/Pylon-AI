"""Live market-residual forecasts (`market_logit`), archived separately in `ledger/market_residual/`.

The model is frozen: coefficients fitted on validation seasons only (`sfi market-validate` writes
`artifacts/market_logit.json`). At issue time each game gets, per target (home win, home ATS cover at the
at-issue spread, over at the at-issue total), the market's de-vigged probability and market_logit's. These
records never enter the independent-forecast ledger or tables.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sfi.data.schema import Game
from sfi.market.dataset import MarketRow
from sfi.market.model import TARGETS, features, market_p, names, outcome

FAMILY = "market_residual"
LABEL = "market-residual forecast (uses market lines as inputs; separate from the independent forecast)"


def fit_artifact(rows: list[MarketRow], seasons: list[int], l2: float, snapshot_id: str) -> dict:
    from sfi.market.model import design, fit
    out = {"model": "market_logit", "version": "0.1.0", "fit_seasons": seasons, "l2": l2, "snapshot_id": snapshot_id,
           "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "targets": {}}
    for t in TARGETS:
        X, y, _ = design([x for x in rows if x.game.season in seasons], t)
        beta = fit(X, y, l2)
        out["targets"][t] = {"terms": ["intercept", "market_logit", *names(t)], "beta": [float(b) for b in beta], "n": len(y)}
    blob = json.dumps(out["targets"], sort_keys=True).encode()
    out["fingerprint"] = hashlib.sha256(blob).hexdigest()[:8]
    return out


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def forecast_records(rows: list[MarketRow], artifact: dict, *, issued_at: datetime, run_id: str,
                     snapshot_id: str) -> list[dict]:
    """One record per game and target, for games kicking off after `issued_at`."""
    out = []
    stamp = issued_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    version = f"{artifact['version']}+{artifact['fingerprint']}"
    for x in rows:
        g = x.game
        if g.scheduled_start_utc <= issued_at:
            continue
        for t in TARGETS:
            pm = market_p(x, t)
            if pm is None:
                continue
            beta = np.asarray(artifact["targets"][t]["beta"])
            z = float(beta @ np.asarray([1.0, _logit(pm), *features(x, t)]))
            p = 1 / (1 + math.exp(-max(min(z, 30), -30)))
            line = {"win": None, "cover": x.spread, "over": x.total}[t]
            prices = {"win": (x.home_ml, x.away_ml), "cover": (x.home_spread_odds, x.away_spread_odds),
                      "over": (x.over_odds, x.under_odds)}[t]
            rec = {"family": FAMILY, "model_name": "market_logit", "model_version": version, "target": t,
                   "game_id": g.game_id, "season": g.season, "week": g.week, "home_team": g.home_team,
                   "away_team": g.away_team, "scheduled_start_utc": g.scheduled_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "prediction_as_of_utc": stamp, "run_id": run_id, "source_snapshot_id": snapshot_id,
                   "line_at_issue": line, "prices_at_issue": list(prices), "p_market": round(pm, 6), "p_model": round(p, 6),
                   "features": dict(zip(names(t), [float(v) for v in features(x, t)])),
                   "lines_treated_as": "at issue (time-stamped by this run)"}
            rec["forecast_id"] = f"market_logit-{version}-{g.game_id}-{t}-{stamp}"
            rec["record_hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
            out.append(rec)
    return out


def append_records(path: Path, records: list[dict]) -> int:
    """Append-only; refuses to rewrite an existing forecast id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                existing[r["forecast_id"]] = r["record_hash"]
    new = []
    for r in records:
        h = existing.get(r["forecast_id"])
        if h is None:
            new.append(r)
        elif h != r["record_hash"]:
            raise RuntimeError(f"{r['forecast_id']} already archived with a different hash")
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for r in new:
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    return len(new)


def grade_outcome(rec: dict, g: Game) -> float | None:
    """Outcome at the line the forecast saw: home win; home cover vs the at-issue spread; over vs the at-issue total."""
    if not g.is_final:
        return None
    m, tot = g.home_score - g.away_score, g.home_score + g.away_score
    if rec["target"] == "win":
        return None if m == 0 else float(m > 0)
    line = rec["line_at_issue"]
    if line is None:
        return None
    val = m if rec["target"] == "cover" else tot
    return None if val == line else float(val > line)


def score(records: list[dict], games: dict[str, Game], n_boot: int = 2000) -> dict:
    """Latest pre-kickoff record per (game, target): Brier of market_logit and of the market at issue."""
    from sfi.evaluation.metrics import block_bootstrap_mean_ci
    latest: dict[tuple, dict] = {}
    for r in records:
        g = games.get(r["game_id"])
        if g is None or r["prediction_as_of_utc"] >= g.scheduled_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"):
            continue
        k = (r["game_id"], r["target"])
        if k not in latest or r["prediction_as_of_utc"] > latest[k]["prediction_as_of_utc"]:
            latest[k] = r
    out = {}
    for t in TARGETS:
        pm, pk, ys, bl, pending = [], [], [], [], 0
        for (gid, tt), r in latest.items():
            if tt != t:
                continue
            g = games[gid]
            y = grade_outcome(r, g)
            if y is None:
                pending += int(not g.is_final)
                continue
            pm.append(r["p_model"]); pk.append(r["p_market"]); ys.append(y); bl.append(f"{g.season}-W{g.week:02d}")
        res = {"n": len(ys), "pending": pending}
        if ys:
            pm_, pk_, y_ = map(np.asarray, (pm, pk, ys))
            d = (pm_ - y_) ** 2 - (pk_ - y_) ** 2
            m, lo, hi = block_bootstrap_mean_ci(d, bl, n_boot)
            res.update(brier_model=float(np.mean((pm_ - y_) ** 2)), brier_market=float(np.mean((pk_ - y_) ** 2)),
                       diff=m, ci=[lo, hi])
        out[t] = res
    return out
