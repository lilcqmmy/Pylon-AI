"""Team-game offensive/defensive aggregates from nflverse play-by-play, and point-in-time team features.

Aggregates are stored as sums (numerator and denominator per metric), so rolling and season-to-date
rates are play-weighted. `TeamHistory` only learns a game's stats when the walk-forward replay hands
it that game's (already known) result, so features at forecast time come from past games only.

Plays counted: pass or rush plays (sacks and scrambles count as pass) with a non-missing EPA,
excluding two-point tries, spikes and kneels.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

SUM_FIELDS = ("plays", "epa", "success", "pass_plays", "pass_epa", "rush_plays", "rush_epa",
              "explosive", "early", "early_pass", "turnovers", "dropbacks", "pressure")

# metric -> (numerator, denominator). Offense: the team's own plays. Defense: opponents' plays
# against the team (so a low defensive EPA/play is good; a high defensive pressure rate is good).
METRICS = {
    "epa_play": ("epa", "plays"),
    "success_rate": ("success", "plays"),
    "pass_epa": ("pass_epa", "pass_plays"),
    "rush_epa": ("rush_epa", "rush_plays"),
    "explosive_rate": ("explosive", "plays"),
    "early_down_pass_rate": ("early_pass", "early"),
    "turnover_rate": ("turnovers", "plays"),
    "pressure_rate": ("pressure", "dropbacks"),  # proxy: sack or QB hit per dropback
}
CACHE_VERSION = "v1"


def aggregate_season(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """{(game_id, posteam): sums} for one season's play-by-play parquet file."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    cols = ["game_id", "posteam", "defteam", "pass", "rush", "epa", "success", "yards_gained", "down",
            "interception", "fumble_lost", "sack", "qb_hit", "qb_dropback", "two_point_attempt",
            "qb_spike", "qb_kneel"]
    t = pq.read_table(path, columns=cols)

    def f(name):  # numeric column with nulls as 0
        return pc.fill_null(t[name].cast("double"), 0.0)

    is_pass, is_rush = pc.equal(f("pass"), 1.0), pc.equal(f("rush"), 1.0)
    keep = pc.and_(
        pc.and_(pc.or_(is_pass, is_rush), pc.is_valid(t["epa"])),
        pc.and_(pc.and_(pc.not_equal(f("two_point_attempt"), 1.0), pc.not_equal(f("qb_spike"), 1.0)),
                pc.and_(pc.not_equal(f("qb_kneel"), 1.0), pc.is_valid(t["posteam"]))),
    )
    yards, down = f("yards_gained"), f("down")
    early = pc.or_(pc.equal(down, 1.0), pc.equal(down, 2.0))
    explosive = pc.or_(pc.and_(is_pass, pc.greater_equal(yards, 20.0)),
                       pc.and_(is_rush, pc.greater_equal(yards, 10.0)))
    dropback = pc.equal(f("qb_dropback"), 1.0)
    pressure = pc.and_(dropback, pc.or_(pc.equal(f("sack"), 1.0), pc.equal(f("qb_hit"), 1.0)))
    epa = f("epa")

    def num(mask):
        return pc.cast(mask, "double")

    import pyarrow as pa
    table = pa.table({
        "game_id": t["game_id"], "posteam": t["posteam"],
        "plays": pa.array([1.0] * len(t)), "epa": epa, "success": f("success"),
        "pass_plays": num(is_pass), "pass_epa": pc.multiply(epa, num(is_pass)),
        "rush_plays": num(is_rush), "rush_epa": pc.multiply(epa, num(is_rush)),
        "explosive": num(explosive), "early": num(early), "early_pass": num(pc.and_(early, is_pass)),
        "turnovers": pc.add(f("interception"), f("fumble_lost")),
        "dropbacks": num(dropback), "pressure": num(pressure),
    }).filter(keep)
    agg = table.group_by(["game_id", "posteam"]).aggregate([(c, "sum") for c in SUM_FIELDS])
    d = agg.to_pydict()
    return {
        (gid, team): {c: float(d[f"{c}_sum"][i]) for c in SUM_FIELDS}
        for i, (gid, team) in enumerate(zip(d["game_id"], d["posteam"]))
    }


def load_team_games(files: dict[int, tuple[Path, dict]], cache_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Aggregate every season, caching per season keyed by the raw file's checksum."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for season, (path, manifest) in sorted(files.items()):
        cache = cache_dir / f"team_games_{season}_{manifest['sha256'][:12]}_{CACHE_VERSION}.csv"
        if cache.exists():
            with open(cache, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    out[(r["game_id"], r["posteam"])] = {c: float(r[c]) for c in SUM_FIELDS}
            continue
        season_rows = aggregate_season(path)
        tmp = cache.with_suffix(".partial")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["game_id", "posteam", *SUM_FIELDS])
            for (gid, team), sums in sorted(season_rows.items()):
                w.writerow([gid, team, *(repr(sums[c]) for c in SUM_FIELDS)])
        tmp.replace(cache)
        out.update(season_rows)
    return out


def _rate(sums: dict[str, float], metric: str) -> float:
    n, d = METRICS[metric]
    return sums[n] / sums[d] if sums.get(d, 0.0) > 0 else math.nan


def _add(acc: dict[str, float], sums: dict[str, float]) -> None:
    for c in SUM_FIELDS:
        acc[c] = acc.get(c, 0.0) + sums[c]


@dataclass
class _Team:
    recent: deque = field(default_factory=lambda: deque(maxlen=4))  # (off_sums, def_sums), crosses seasons
    season_off: dict = field(default_factory=dict)
    season_def: dict = field(default_factory=dict)
    prior_off: dict = field(default_factory=dict)  # previous season totals (early-season fallback)
    prior_def: dict = field(default_factory=dict)
    games_this_season: int = 0


class TeamHistory:
    """Point-in-time per-team aggregates. Learns only through `observe_game`."""

    def __init__(self, metrics: list[str] | None = None, windows: tuple[str, ...] = ("r4", "std")):
        self.metrics = list(metrics or METRICS)
        self.windows = windows
        self.teams: dict[str, _Team] = defaultdict(_Team)

    def start_season(self) -> None:
        for t in self.teams.values():
            t.prior_off, t.prior_def = t.season_off, t.season_def
            t.season_off, t.season_def = {}, {}
            t.games_this_season = 0

    def observe_game(self, team: str, off: dict[str, float], deff: dict[str, float]) -> None:
        t = self.teams[team]
        t.recent.append((off, deff))
        _add(t.season_off, off)
        _add(t.season_def, deff)
        t.games_this_season += 1

    def features(self, team: str) -> dict[str, float]:
        t = self.teams[team]
        r_off, r_def = {}, {}
        for off, deff in t.recent:
            _add(r_off, off)
            _add(r_def, deff)
        s_off = t.season_off if t.games_this_season else t.prior_off
        s_def = t.season_def if t.games_this_season else t.prior_def
        src = {("off", "r4"): r_off, ("def", "r4"): r_def, ("off", "std"): s_off, ("def", "std"): s_def}
        return {
            f"{side}_{m}_{w}": _rate(src[(side, w)], m) if src[(side, w)] else math.nan
            for side in ("off", "def") for w in self.windows for m in self.metrics
        }

    def feature_names(self) -> list[str]:
        return [f"{side}_{m}_{w}" for side in ("off", "def") for w in self.windows for m in self.metrics]
