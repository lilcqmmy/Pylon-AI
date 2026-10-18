# Reproduces docs/data_sources/nflverse_availability.md. Run: .venv/Scripts/python.exe -I scripts/analysis/<name>.py "<project root>"
"""How much does QB availability explain beyond Elo? Validation seasons only (<= 2019). Read-only."""
import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from sfi.data.sources import nflverse as N  # noqa: E402
from sfi.evaluation import metrics as M  # noqa: E402
from sfi.evaluation.walkforward import iter_walkforward  # noqa: E402
from sfi.models.elo import EloModel, EloParams  # noqa: E402
from sfi.models.pbp_logit import fit_logistic_l2  # noqa: E402

LAST = 2019  # holdout 2020-2025 stays sealed: no outcomes after 2019 are examined
folder = N.latest_snapshot(ROOT / "data/raw/nflverse/schedules")
snap = N.load_snapshot(folder, min_season=2000, known_at_hours=5)
games = [g for g in snap.games if g.season <= LAST]
raw = {r["game_id"]: r for r in csv.DictReader(open(folder / "games.csv", encoding="utf-8"))}
params = EloParams(k_factor=20, home_field_elo=48, season_regression=0.5, margin_of_victory=True)
p_elo = {f.game.game_id: f.p_home_win for f in iter_walkforward(games, EloModel(params), with_inputs=False)}

# Previous starter per team (by franchise, across seasons) and actual starter (hindsight).
seq = defaultdict(list)
for g in games:
    r = raw[g.game_id]
    for side, key, qb in (("home", g.home_key, r["home_qb_id"]), ("away", g.away_key, r["away_qb_id"])):
        seq[key].append((g.scheduled_start_utc, g.game_id, side, qb if qb not in ("", "NA") else None))
actual, prev = {}, {}
for key, lst in seq.items():
    lst.sort()
    for i, (_, gid, side, qb) in enumerate(lst):
        actual[(gid, side)] = qb
        prev[(gid, side)] = lst[i - 1][3] if i else None

# Injury report status of the previous starter for this game (2010+; date_modified before kickoff).
status = {}
for s in range(2010, LAST + 1):
    fs = sorted(glob.glob(str(ROOT / f"data/raw/nflverse/injuries/{s}/*/injuries_{s}.parquet")))
    t = pq.read_table(fs[-1]).to_pydict()
    for team, week, gsis, st, dm in zip(t["team"], t["week"], t["gsis_id"], t["report_status"], t["date_modified"]):
        status[(s, int(week), N.FRANCHISE.get(team, team), gsis)] = (st, dm)

rows = []
for g in games:
    if not g.is_final or g.home_outcome == 0.5 or g.season < 2001:
        continue
    feats = {}
    for side, key in (("home", g.home_key), ("away", g.away_key)):
        a, p = actual[(g.game_id, side)], prev[(g.game_id, side)]
        feats[f"{side}_changed"] = float(a is not None and p is not None and a != p)
        out = None
        if g.season >= 2010 and p is not None:
            st = status.get((g.season, g.week, key, p))
            if st and st[1] is not None and st[1] < g.scheduled_start_utc:
                out = st[0]
        feats[f"{side}_prev_out"] = float(out in ("Out", "Doubtful"))
        feats[f"{side}_prev_q"] = float(out == "Questionable")
    rows.append((g, feats))


def evaluate(names, label, seasons_train, seasons_test, flag_rows=None):
    tr = [(g, f) for g, f in rows if g.season in seasons_train]
    te = [(g, f) for g, f in rows if g.season in seasons_test]

    def X(rs):
        return np.array([[1.0, math.log(p_elo[g.game_id] / (1 - p_elo[g.game_id]))] + [f[n] for n in names] for g, f in rs])

    y_tr = np.array([g.home_outcome for g, _ in tr])
    beta = fit_logistic_l2(X(tr), y_tr, l2=1.0)
    p_new = 1 / (1 + np.exp(-(X(te) @ beta)))
    y = np.array([g.home_outcome for g, _ in te])
    pe = np.array([p_elo[g.game_id] for g, _ in te])
    d = M.brier_terms(p_new, y) - M.brier_terms(pe, y)
    m, lo, hi = M.block_bootstrap_mean_ci(d, [f"{g.season}-{g.week}" for g, _ in te], 2000)
    flagged = [i for i, (g, f) in enumerate(te) if any(f[n] for n in names)]
    print(f"{label}: coefs {dict(zip(['int', 'elo_logit', *names], np.round(beta, 3)))}")
    print(f"   test {min(seasons_test)}-{max(seasons_test)}: {len(te)} games, {len(flagged)} flagged; "
          f"Brier diff vs Elo {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    if flagged:
        fd = d[flagged]
        print(f"   on flagged games only: Brier diff {fd.mean():+.4f} (n={len(flagged)})")


print("Base rates (2001-2019):")
for n in ("home_changed", "away_changed"):
    k = sum(f[n] for _, f in rows)
    print(f"  {n}: {int(k)} of {len(rows)} games ({k / len(rows):.1%})")
for n in ("home_prev_out", "away_prev_out"):
    k = sum(f[n] for g, f in rows if g.season >= 2010)
    tot = sum(1 for g, _ in rows if g.season >= 2010)
    print(f"  {n} (2010+): {int(k)} of {tot}")

print("\nResidual check: mean(outcome - Elo prob) by actual-starter change (hindsight)")
for cond, lab in ((lambda f: f["home_changed"] and not f["away_changed"], "home QB changed only"),
                  (lambda f: f["away_changed"] and not f["home_changed"], "away QB changed only"),
                  (lambda f: not f["home_changed"] and not f["away_changed"], "neither changed")):
    sel = [(g, f) for g, f in rows if cond(f)]
    r = np.mean([g.home_outcome - p_elo[g.game_id] for g, _ in sel])
    print(f"  {lab}: n={len(sel)}, mean residual {r:+.3f}")

print()
evaluate(["home_changed", "away_changed"], "UPPER BOUND (hindsight: actual starter differs from last game)",
         set(range(2001, 2011)), set(range(2011, 2020)))
evaluate(["home_prev_out", "away_prev_out", "home_prev_q", "away_prev_q"],
         "PREGAME PROXY (last game's starter Out/Doubtful/Questionable on injury report)",
         set(range(2010, 2015)), set(range(2015, 2020)))


# ---- QB quality upper bound: prior shrunk EPA/dropback of actual vs previous starter (hindsight starter)
import pyarrow.compute as pc  # noqa: E402

qb_games = defaultdict(list)  # qb -> [(kickoff, epa_sum, dropbacks)]
kick = {g.game_id: g.scheduled_start_utc for g in games}
for s in range(2000, LAST + 1):
    fs = sorted(glob.glob(str(ROOT / f"data/raw/nflverse/pbp/{s}/*/play_by_play_{s}.parquet")))
    t = pq.read_table(fs[-1], columns=["game_id", "id", "qb_epa", "qb_dropback"])
    t = t.filter(pc.and_(pc.equal(pc.fill_null(t["qb_dropback"], 0.0), 1.0), pc.is_valid(t["id"])))
    agg = t.group_by(["game_id", "id"]).aggregate([("qb_epa", "sum"), ("qb_dropback", "sum")]).to_pydict()
    for gid, qb, e, n in zip(agg["game_id"], agg["id"], agg["qb_epa_sum"], agg["qb_dropback_sum"]):
        if gid in kick and e is not None:
            qb_games[qb].append((kick[gid], e, n))
for v in qb_games.values():
    v.sort()
LEAGUE, K = 0.0, 300.0  # shrink toward replacement-ish 0.0 EPA/db with 300 dropbacks of prior (untuned)


def rating(qb, before):
    if qb is None:
        return None
    e = n = 0.0
    for t0, ee, nn in qb_games.get(qb, []):
        if t0 >= before:
            break
        e += ee
        n += nn
    return (e + K * LEAGUE) / (n + K)


for g, f in rows:
    for side in ("home", "away"):
        a, p = actual[(g.game_id, side)], prev[(g.game_id, side)]
        ra, rp = rating(a, g.scheduled_start_utc), rating(p, g.scheduled_start_utc)
        f[f"{side}_qb_delta"] = (ra - rp) if (ra is not None and rp is not None and a != p) else 0.0
    f["qb_delta_diff"] = f["home_qb_delta"] - f["away_qb_delta"]

print()
evaluate(["qb_delta_diff"], "UPPER BOUND with QB quality (hindsight starter; prior EPA/dropback change, home - away)",
         set(range(2001, 2011)), set(range(2011, 2020)))
evaluate(["qb_delta_diff", "home_changed", "away_changed"], "UPPER BOUND quality + change flags",
         set(range(2001, 2011)), set(range(2011, 2020)))
