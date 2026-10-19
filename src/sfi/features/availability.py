"""Expected starters per team and position group, from nflverse depth charts (point-in-time where possible).

- Through 2024: weekly charts (no timestamp). The chart for the game's own (season, week) is used; its
  publication time is undocumented (analysis showed it is not hindsight-filled but lags real decisions).
- From 2025: daily snapshots with `dt`; the latest snapshot at or before the as-of time is used.

Position groups and importance weights are fixed by reasoning, not fitted (docs/challenger_avail_qbadj.md).
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sfi.data.sources.nflverse import FRANCHISE

GROUPS = {
    "QB": {"QB"},
    "OL": {"T", "G", "C", "LT", "LG", "RT", "RG", "OT", "OG", "OL"},
    "SECONDARY": {"CB", "S", "SS", "FS", "DB", "NB", "LCB", "RCB"},
    "PASS_RUSH": {"DE", "DT", "NT", "OLB", "LDE", "RDE", "LDT", "RDT", "EDGE", "DL", "LOLB", "ROLB"},
    "SKILL": {"WR", "TE", "RB", "HB"},
}
WEIGHTS = {"QB": 3.0, "OL": 1.0, "SECONDARY": 1.0, "PASS_RUSH": 0.7, "SKILL": 0.5}
POS_TO_GROUP = {p: g for g, ps in GROUPS.items() for p in ps}

Starters = dict[str, str]  # gsis_id -> group


class DepthCharts:
    def __init__(self) -> None:
        self.weekly: dict[tuple[int, int, str], Starters] = {}
        self.daily: dict[str, tuple[list[datetime], list[Starters]]] = {}
        self.qb_weekly: dict[tuple[int, int, str], list[str]] = {}            # QB ids in depth order
        self.qb_daily: dict[str, tuple[list[datetime], list[list[str]]]] = {}

    def qb_order(self, season: int, week: int, team: str, as_of: datetime | None) -> list[str] | None:
        """QBs in depth order at forecast time (same rules as `starters`), or None."""
        if (season, week, team) in self.qb_weekly:
            return self.qb_weekly[(season, week, team)]
        if team in self.qb_daily and as_of is not None:
            times, orders = self.qb_daily[team]
            i = bisect.bisect_right(times, as_of) - 1
            if i >= 0 and times[i].year >= season - 1 and (as_of - times[i]).days <= 14:
                return orders[i]
        return None

    def starters(self, season: int, week: int, team: str, as_of: datetime | None) -> Starters | None:
        """Expected starters for a team at forecast time, or None if no chart is available."""
        if (season, week, team) in self.weekly:
            return self.weekly[(season, week, team)]
        if team in self.daily and as_of is not None:
            times, charts = self.daily[team]
            i = bisect.bisect_right(times, as_of) - 1
            if i >= 0 and times[i].year >= season - 1 and (as_of - times[i]).days <= 14:
                return charts[i]
        return None


def load_depth_charts(files: dict[int, tuple[Path, dict]]) -> DepthCharts:
    import pyarrow.parquet as pq

    dc = DepthCharts()
    daily: dict[str, dict[datetime, Starters]] = defaultdict(dict)
    qb_daily: dict[str, dict[datetime, list]] = defaultdict(dict)
    qb_weekly: dict[tuple[int, int, str], list] = defaultdict(list)
    for season, (path, _) in sorted(files.items()):
        t = pq.read_table(path)
        d = t.to_pydict()
        if "dt" in d:  # 2025+ daily snapshots
            for dt, team, gsis, pos, rank in zip(d["dt"], d["team"], d["gsis_id"], d["pos_abb"], d["pos_rank"]):
                if pos == "QB" and gsis and rank is not None:
                    when = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    qb_daily[FRANCHISE.get(team, team)].setdefault(when, []).append((int(rank), gsis))
                grp = POS_TO_GROUP.get(pos)
                if rank != 1 or grp is None or not gsis:
                    continue
                when = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                daily[FRANCHISE.get(team, team)].setdefault(when, {}).setdefault(gsis, grp)
        else:  # weekly charts
            for s, club, week, team_rank, form, pos, gsis in zip(d["season"], d["club_code"], d["week"], d["depth_team"],
                                                                 d["formation"], d["position"], d["gsis_id"]):
                if pos == "QB" and gsis and week is not None and form != "Special Teams":
                    try:
                        qb_weekly[(int(s), int(week), FRANCHISE.get(club, club))].append((int(team_rank), gsis))
                    except (TypeError, ValueError):
                        pass
                grp = POS_TO_GROUP.get(pos)
                if str(team_rank) != "1" or grp is None or not gsis or week is None or form == "Special Teams":
                    continue
                key = (int(s), int(week), FRANCHISE.get(club, club))
                dc.weekly.setdefault(key, {}).setdefault(gsis, grp)
    for team, snaps in daily.items():
        times = sorted(snaps)
        dc.daily[team] = (times, [snaps[t] for t in times])

    def ordered(pairs):
        seen, out = set(), []
        for _, pid in sorted(pairs):
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
        return out

    dc.qb_weekly = {k: ordered(v) for k, v in qb_weekly.items()}
    for team, snaps in qb_daily.items():
        times = sorted(snaps)
        dc.qb_daily[team] = (times, [ordered(snaps[t]) for t in times])
    return dc
