# Market-disagreement study, 2006-2019 (hypothesis generator only).
# Run: .venv/Scripts/python.exe -I scripts/analysis/market_disagreement.py "<project root>"
"""Buckets validation games by d = elo_tuned P(home) - market P(home) (de-vigged moneyline in the snapshot;
recording time undocumented, likely near kickoff). Per bucket: games, Elo Brier, market Brier, and Elo's
paper return betting 1 unit on the side Elo prefers (home if d > 0, else away) at the snapshot moneyline,
with a week-block bootstrap CI. Retrospective and leakage-prone in timing: not evidence of an edge."""

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

from sfi.data.sources import nflverse as N  # noqa: E402
from sfi.evaluation.metrics import block_bootstrap_mean_ci  # noqa: E402
from sfi.picks import american_profit  # noqa: E402

folder = N.latest_snapshot(ROOT / "data/raw/nflverse/schedules")
snap = N.load_snapshot(folder, min_season=2000, known_at_hours=5)
games = {g.game_id: g for g in snap.games}
sid = "sha256:" + json.loads((folder / "manifest.json").read_text())["sha256"][:16]
elo = {}
with open(ROOT / "archive/backtest_forecasts.jsonl", encoding="utf-8") as fh:
    for line in fh:
        if '"model_name":"elo_tuned"' not in line or sid not in line:
            continue
        r = json.loads(line)
        if 2006 <= r["season"] <= 2019:
            elo[r["game_id"]] = r["distribution"]["p_home_win"]

EDGES = [-np.inf, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, np.inf]
LABELS = ["≤ −0.15", "−0.15 to −0.10", "−0.10 to −0.05", "−0.05 to 0", "0 to 0.05", "0.05 to 0.10", "0.10 to 0.15", "≥ 0.15"]
rows = {i: {"y": [], "pe": [], "pm": [], "ret": [], "blk": []} for i in range(len(LABELS))}
for gid, pe in elo.items():
    g = games[gid]
    ln = snap.lines.get(gid, {})
    pm = snap.market_p_home.get(gid)
    if pm is None or not g.is_final or "home_ml" not in ln or "away_ml" not in ln:
        continue
    d = pe - pm
    b = int(np.searchsorted(EDGES, d, side="right") - 1)
    b = min(max(b, 0), len(LABELS) - 1)
    side_home = d > 0
    odds = ln["home_ml"] if side_home else ln["away_ml"]
    if g.home_outcome == 0.5:
        ret = 0.0
    else:
        won = (g.home_outcome == 1.0) == side_home
        ret = american_profit(odds, 1.0) if won else -1.0
    r = rows[b]
    r["ret"].append(ret)
    r["blk"].append(f"{g.season}-{g.week}")
    if g.home_outcome != 0.5:
        r["y"].append(g.home_outcome)
        r["pe"].append(pe)
        r["pm"].append(pm)

out = []
for i, lab in enumerate(LABELS):
    r = rows[i]
    if not r["ret"]:
        continue
    y, pe, pm = map(np.asarray, (r["y"], r["pe"], r["pm"]))
    m, lo, hi = block_bootstrap_mean_ci(r["ret"], r["blk"], 4000)
    out.append({"bucket": lab, "games": len(r["ret"]), "elo_brier": float(np.mean((pe - y) ** 2)),
                "market_brier": float(np.mean((pm - y) ** 2)),
                "return_per_unit": m, "return_ci95": [lo, hi]})
tot = sum(o["games"] for o in out)
print(json.dumps({"games": tot, "buckets": out}, indent=1))

# Pooled paper return for the live paper-pick rule thresholds (|d| >= x), side = Elo's preferred side.
pooled = {}
for thr in (0.03, 0.05, 0.10):
    rets, blks = [], []
    for gid, pe in elo.items():
        g = games[gid]
        ln = snap.lines.get(gid, {})
        pm = snap.market_p_home.get(gid)
        if pm is None or not g.is_final or "home_ml" not in ln or "away_ml" not in ln or abs(pe - pm) < thr:
            continue
        side_home = pe > pm
        odds = ln["home_ml"] if side_home else ln["away_ml"]
        rets.append(0.0 if g.home_outcome == 0.5 else (american_profit(odds, 1.0) if (g.home_outcome == 1.0) == side_home else -1.0))
        blks.append(f"{g.season}-{g.week}")
    m, lo, hi = block_bootstrap_mean_ci(rets, blks, 4000)
    pooled[str(thr)] = {"games": len(rets), "return_per_unit": m, "return_ci95": [lo, hi]}
print(json.dumps({"pooled": pooled}, indent=1))
