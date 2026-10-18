"""Inputs for the QB-availability adjustment (see docs/data_sources/nflverse_availability.md).

- QB game stats: per game, each passer's summed `qb_epa` and dropbacks (play-by-play).
- Starters: each team's actual starting QB per game (schedules). This is hindsight for the game itself,
  so models may read it only for games whose results have been released.
- Injury rows (QBs by default), each with an explicit known-at time: `date_modified` when present and before
  kickoff; rows modified at/after kickoff are dropped; when the timestamp is missing (2009, 2025+) the
  row is treated as known `missing_hours` before kickoff.
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sfi.data.schema import Game
from sfi.data.sources.nflverse import FRANCHISE, NA

CACHE_VERSION = "v1"


@dataclass(frozen=True)
class InjuryRow:
    player: str
    status: str | None
    known_at: datetime
    timestamp_source: str  # "date_modified" | "policy"


@dataclass
class InjuryLoad:
    rows: dict[tuple[int, int, str], list[InjuryRow]] = field(default_factory=dict)
    dropped_after_kickoff: int = 0
    unmatched: int = 0
    policy_rows: int = 0


def aggregate_qb_season(path: Path) -> dict[str, list[tuple[str, str, float, float]]]:
    """{game_id: [(qb_id, posteam, qb_epa_sum, dropbacks)]} for one play-by-play season."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["game_id", "id", "posteam", "qb_epa", "qb_dropback"])
    keep = pc.and_(pc.equal(pc.fill_null(t["qb_dropback"], 0.0), 1.0),
                   pc.and_(pc.is_valid(t["id"]), pc.is_valid(t["posteam"])))
    t = t.filter(keep)
    t = t.set_column(t.column_names.index("qb_epa"), "qb_epa", pc.fill_null(t["qb_epa"], 0.0))
    d = t.group_by(["game_id", "id", "posteam"]).aggregate([("qb_epa", "sum"), ("qb_dropback", "sum")]).to_pydict()
    out: dict[str, list] = defaultdict(list)
    for gid, qb, team, e, n in zip(d["game_id"], d["id"], d["posteam"], d["qb_epa_sum"], d["qb_dropback_sum"]):
        out[gid].append((qb, FRANCHISE.get(team, team), float(e), float(n)))
    return dict(out)


def load_qb_games(files: dict[int, tuple[Path, dict]], cache_dir: Path) -> dict[str, list[tuple[str, str, float, float]]]:
    out: dict[str, list] = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for season, (path, manifest) in sorted(files.items()):
        cache = cache_dir / f"qb_games_{season}_{manifest['sha256'][:12]}_{CACHE_VERSION}.csv"
        if not cache.exists():
            rows = aggregate_qb_season(path)
            tmp = cache.with_suffix(".partial")
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(["game_id", "qb", "team", "qb_epa", "dropbacks"])
                for gid in sorted(rows):
                    for qb, team, e, n in sorted(rows[gid]):
                        w.writerow([gid, qb, team, repr(e), repr(n)])
            tmp.replace(cache)
        with open(cache, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out.setdefault(r["game_id"], []).append((r["qb"], r["team"], float(r["qb_epa"]), float(r["dropbacks"])))
    return out


def load_starters(schedule_csv: Path, games: list[Game]) -> dict[tuple[str, str], str]:
    """{(game_id, franchise): starting QB id} from the schedule snapshot's (hindsight) starter columns."""
    by_id = {g.game_id: g for g in games}
    out = {}
    with open(schedule_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            g = by_id.get(r["game_id"])
            if g is None:
                continue
            for key, col in ((g.home_key, "home_qb_id"), (g.away_key, "away_qb_id")):
                if r.get(col) not in (None, *NA):
                    out[(g.game_id, key)] = r[col]
    return out


def load_injuries(files: dict[int, tuple[Path, dict]], games: list[Game], missing_hours: float = 24.0,
                  positions: set[str] | None = frozenset({"QB"})) -> InjuryLoad:
    """Injury rows for `positions` (None = all positions)."""
    import pyarrow.parquet as pq

    kickoff = {}
    for g in games:
        for key in (g.home_key, g.away_key):
            kickoff[(g.season, g.week, key)] = g.scheduled_start_utc
    res = InjuryLoad()
    for season, (path, _) in sorted(files.items()):
        t = pq.read_table(path)
        d = t.to_pydict()
        dm = d.get("date_modified") or [None] * t.num_rows
        for team, week, gsis, pos, st, mod in zip(d["team"], d["week"], d["gsis_id"], d["position"], d["report_status"], dm):
            if (positions is not None and pos not in positions) or gsis is None or week is None:
                continue
            key = (int(season), int(week), FRANCHISE.get(team, team))
            ko = kickoff.get(key)
            if ko is None:
                res.unmatched += 1
                continue
            if mod is not None:
                mod = mod if isinstance(mod, datetime) else datetime.fromisoformat(str(mod))
                if mod.tzinfo is None:
                    mod = mod.replace(tzinfo=timezone.utc)
                if mod >= ko:
                    res.dropped_after_kickoff += 1
                    continue
                row = InjuryRow(gsis, st, mod, "date_modified")
            else:
                row = InjuryRow(gsis, st, ko - timedelta(hours=missing_hours), "policy")
                res.policy_rows += 1
            res.rows.setdefault(key, []).append(row)
    return res
