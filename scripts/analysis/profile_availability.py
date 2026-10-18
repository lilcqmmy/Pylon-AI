# Reproduces docs/data_sources/nflverse_availability.md. Run: .venv/Scripts/python.exe -I scripts/analysis/<name>.py "<project root>"
"""Profile injuries + depth charts timing against schedules. Read-only."""
import glob
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
import pyarrow.parquet as pq  # noqa: E402
from sfi.data.sources import nflverse as N  # noqa: E402

snap = N.load_snapshot(N.latest_snapshot(ROOT / "data/raw/nflverse/schedules"), min_season=2000, known_at_hours=5)
rows = {}
import csv  # noqa: E402
folder = N.latest_snapshot(ROOT / "data/raw/nflverse/schedules")
raw = {r["game_id"]: r for r in csv.DictReader(open(folder / "games.csv", encoding="utf-8"))}
games = {g.game_id: g for g in snap.games}
# (season, week, team_code) -> game, using schedule team codes and franchise keys
by_team_week = {}
for g in snap.games:
    for code in {g.home_team, g.home_key}:
        by_team_week[(g.season, g.week, code)] = g
    for code in {g.away_team, g.away_key}:
        by_team_week[(g.season, g.week, code)] = g


def latest(kind, season):
    fs = sorted(glob.glob(str(ROOT / f"data/raw/nflverse/{kind}/{season}/*/{kind}_{season}.parquet")))
    return fs[-1] if fs else None


print("== injuries: date_modified coverage and timing vs kickoff")
for s in range(2009, 2027):
    t = pq.read_table(latest("injuries", s)).to_pydict()
    n = len(t["season"])
    dm = t.get("date_modified")
    nn = sum(v is not None for v in dm) if dm else 0
    leads, unmatched, teams_missing = [], 0, Counter()
    if dm:
        for team, week, d in zip(t["team"], t["week"], dm):
            g = by_team_week.get((s, int(week), team))
            if g is None:
                unmatched += 1
                teams_missing[team] += 1
                continue
            if d is not None:
                leads.append((g.scheduled_start_utc - d).total_seconds() / 3600)
    else:
        for team, week in zip(t["team"], t["week"]):
            if by_team_week.get((s, int(week), team)) is None:
                unmatched += 1
                teams_missing[team] += 1
    leads.sort()
    q = (lambda p: round(leads[int(p * (len(leads) - 1))], 1)) if leads else (lambda p: None)
    after = sum(1 for x in leads if x < 0)
    print(f"{s}: rows {n}, date_modified {nn}/{n}, unmatched team-weeks {unmatched} {dict(teams_missing.most_common(3))}; "
          f"hours before kickoff p5/p50/p95 = {q(.05)}/{q(.5)}/{q(.95)}; modified AFTER kickoff: {after}")

print("\n== QB1 on pre-2025 weekly depth charts vs actual starters (schedule qb ids)")
start_of = {}  # (season, franchise, week) -> starter gsis
seq = defaultdict(list)
for gid, r in raw.items():
    g = games.get(gid)
    if g is None or not g.is_final:
        continue
    for key, qb in ((g.home_key, r["home_qb_id"]), (g.away_key, r["away_qb_id"])):
        if qb not in ("", "NA"):
            seq[(g.season, key)].append((g.scheduled_start_utc, g.week, qb))
prev_starter = {}
for (season, key), lst in seq.items():
    lst.sort()
    for i, (_, week, qb) in enumerate(lst):
        start_of[(season, key, week)] = qb
        prev_starter[(season, key, week)] = lst[i - 1][2] if i else None

for s in (2001, 2005, 2010, 2015, 2019, 2023, 2024):
    t = pq.read_table(latest("depth_charts", s)).to_pydict()
    qb1 = {}
    for season, club, week, team_rank, pos, dpos, gsis in zip(t["season"], t["club_code"], t["week"], t["depth_team"],
                                                              t["position"], t["depth_position"], t["gsis_id"]):
        if pos == "QB" and str(team_rank) == "1" and week is not None:
            key = N.FRANCHISE.get(club, club)
            qb1.setdefault((s, key, int(week)), gsis)
    same = changed = changed_dc_new = changed_dc_old = 0
    total = matched = 0
    for k, starter in start_of.items():
        if k[0] != s or k not in qb1:
            continue
        total += 1
        matched += qb1[k] == starter
        p = prev_starter.get(k)
        if p is not None and p != starter:
            changed += 1
            changed_dc_new += qb1[k] == starter
            changed_dc_old += qb1[k] == p
    print(f"{s}: depth QB1 == actual starter in {matched}/{total} team-games; on {changed} starter changes, "
          f"depth QB1 matched the NEW starter {changed_dc_new} times, the PREVIOUS starter {changed_dc_old} times")

print("\n== 2025+ timestamped depth charts")
for s in (2025, 2026):
    t = pq.read_table(latest("depth_charts", s), columns=["dt", "team", "pos_abb", "pos_rank", "gsis_id"]).to_pydict()
    dts = sorted(set(t["dt"]))
    print(f"{s}: {len(dts)} distinct snapshots, first {dts[0]}, last {dts[-1]}")
    from datetime import datetime, timezone
    times = [datetime.strptime(d, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) for d in dts]
    gaps = sorted((b - a).total_seconds() / 3600 for a, b in zip(times, times[1:]))
    print(f"   gap hours p10/p50/p90: {gaps[len(gaps)//10]:.1f}/{gaps[len(gaps)//2]:.1f}/{gaps[9*len(gaps)//10]:.1f}")
    # QB1 at latest snapshot before kickoff - 1h vs actual starter
    qb1_by = defaultdict(dict)  # team -> dt -> gsis
    for dt, team, pos, rank, gsis in zip(t["dt"], t["team"], t["pos_abb"], t["pos_rank"], t["gsis_id"]):
        if pos == "QB" and rank == 1:
            qb1_by[N.FRANCHISE.get(team, team)].setdefault(dt, gsis)
    tot = ok = changed = ch_new = ch_old = nosnap = 0
    for (season, key, week), starter in start_of.items():
        if season != s:
            continue
        g = by_team_week[(season, week, key)]
        cutoff = (g.scheduled_start_utc - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        snaps = [d for d in qb1_by.get(key, {}) if d <= cutoff]
        if not snaps:
            nosnap += 1
            continue
        q = qb1_by[key][max(snaps)]
        tot += 1
        ok += q == starter
        p = prev_starter.get((season, key, week))
        if p is not None and p != starter:
            changed += 1
            ch_new += q == starter
            ch_old += q == p
    print(f"   QB1 at last snapshot >=1h before kickoff == actual starter: {ok}/{tot} (no snapshot: {nosnap}); "
          f"on {changed} starter changes: matched new {ch_new}, previous {ch_old}")
