# Minimum detectable effect for paired-Brier tests vs elo_tuned (validation 2010-2019).
# Run: .venv/Scripts/python.exe -I scripts/analysis/mde.py "<project root>"
"""For each challenger already backtested, take the per-game paired Brier differences vs elo_tuned on
2010-2019 (and its pre-specified subset), estimate the standard error of their mean with a week-block
bootstrap, and convert it to a minimum detectable effect (MDE): the smallest true mean improvement that a
two-sided test at level alpha detects with 80% power, MDE = (z_{1-alpha/2} + z_{0.80}) x SE.

The SE depends on how much a challenger's forecasts differ from elo_tuned's, so it is measured per
comparison rather than assumed. Projections to prospective sample sizes assume SE scales as 1/sqrt(n).
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

from sfi.data.sources import nflverse as N  # noqa: E402

Z = {0.95: 1.959964, 1 - 0.05 / 3: 2.393980, 1 - 0.05 / 4: 2.497705}
Z_POWER = 0.841621  # 80% power
TARGET = 0.002

folder = N.latest_snapshot(ROOT / "data/raw/nflverse/schedules")
snap = N.load_snapshot(folder, min_season=2000, known_at_hours=5)
games = {g.game_id: g for g in snap.games}
starters = {}
for r in csv.DictReader(open(folder / "games.csv", encoding="utf-8")):
    g = games.get(r["game_id"])
    if g:
        starters[(g.game_id, g.home_key)] = r["home_qb_id"]
        starters[(g.game_id, g.away_key)] = r["away_qb_id"]

sid = "sha256:" + json.loads((folder / "manifest.json").read_text())["sha256"][:16]
want = {"elo_tuned", "elo_hfa", "elo_avail", "elo_qbadj", "elo_combo", "elo_sched"}
recs = defaultdict(dict)
with open(ROOT / "archive/backtest_forecasts.jsonl", encoding="utf-8") as fh:
    for line in fh:
        if '"source_snapshot_id":"' + sid not in line:
            continue
        r = json.loads(line)
        if r["model_name"] in want and 2010 <= r["season"] <= 2019:
            recs[r["model_name"]][r["game_id"]] = r


def subset_pred(name):
    if name == "elo_avail":
        return "either team 2+ starters out", lambda r: max(r["inputs"]["home"]["n_out"], r["inputs"]["away"]["n_out"]) >= 2
    if name == "elo_qbadj":
        def changed(r):
            g = games[r["game_id"]]
            return any(starters.get((g.game_id, k)) not in (None, "", "NA", (r["inputs"].get(s) or {}).get("prev_starter"))
                       for s, k in (("home_qb", g.home_key), ("away_qb", g.away_key)))
        return "a team's starting QB changed", changed
    return None, None


def block_se(d, blocks, n_boot=4000, seed=0):
    d, blocks = np.asarray(d), np.asarray(blocks)
    uniq, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    cnt = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(boot.std(ddof=1))


rows = []
base = recs["elo_tuned"]
for name in sorted(want - {"elo_tuned"}):
    if name not in recs:
        continue
    desc, pred = subset_pred(name)
    for label, filt in (("all games", None), (desc, pred)):
        if label is None:
            continue
        d, blocks = [], []
        for gid, r in recs[name].items():
            g = games[gid]
            if gid not in base or not g.is_final or g.home_outcome == 0.5:
                continue
            if filt is not None and not filt(r):
                continue
            y = g.home_outcome
            d.append((r["distribution"]["p_home_win"] - y) ** 2 - (base[gid]["distribution"]["p_home_win"] - y) ** 2)
            blocks.append(f"{g.season}-{g.week}")
        if not d:
            continue
        n = len(d)
        se = block_se(d, blocks)
        sd_game = float(np.std(d, ddof=1))
        mde = {lvl: (z + Z_POWER) * se for lvl, z in Z.items()}
        n_needed = {lvl: n * (mde[lvl] / TARGET) ** 2 for lvl in Z}
        rows.append({"model": name, "subset": label, "n": n, "mean_diff": float(np.mean(d)), "se": se, "sd_per_game": sd_game,
                     "mde95": mde[0.95], "mde_bonf3": mde[1 - 0.05 / 3], "mde_bonf4": mde[1 - 0.05 / 4],
                     "mde95_at_500": mde[0.95] * math.sqrt(n / 500), "mde95_at_1000": mde[0.95] * math.sqrt(n / 1000),
                     "games_needed_0.002_95": n_needed[0.95], "games_needed_0.002_bonf3": n_needed[1 - 0.05 / 3]})

print(json.dumps(rows, indent=1))
