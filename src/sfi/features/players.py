"""Player-game yardage from nflverse play-by-play, for player stat distributions.

Per (game_id, player_id, stat) with stat in {"pass", "rush", "rec"}: opportunities (pass attempts,
carries, targets) and yards. Definitions (two-point tries excluded):
- pass: plays with a passer and a completion, incompletion or interception; yards = `passing_yards`
  (sacks are not attempts and their yardage is not passing yardage);
- rush: plays with a rusher (QB scrambles included); yards = `rushing_yards`;
- rec: plays with a targeted receiver (completion, incompletion or interception); yards = `receiving_yards`.
Lateral yardage is not included, so totals can differ slightly from official statistics.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

STATS = ("pass", "rush", "rec")
CACHE_VERSION = "v1"

PlayerGames = dict[str, list[tuple[str, str, str, float, float]]]  # game_id -> [(player, team, stat, opp, yards)]


def aggregate_players_season(path: Path) -> PlayerGames:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from sfi.data.sources.nflverse import FRANCHISE

    cols = ["game_id", "posteam", "passer_player_id", "rusher_player_id", "receiver_player_id", "passing_yards",
            "rushing_yards", "receiving_yards", "complete_pass", "incomplete_pass", "interception", "two_point_attempt"]
    t = pq.read_table(path, columns=cols)

    def f(c):
        return pc.fill_null(t[c].cast("double"), 0.0)

    not_2pt = pc.not_equal(f("two_point_attempt"), 1.0)
    is_target = pc.or_(pc.or_(pc.equal(f("complete_pass"), 1.0), pc.equal(f("incomplete_pass"), 1.0)),
                       pc.equal(f("interception"), 1.0))
    specs = {
        "pass": ("passer_player_id", pc.and_(is_target, not_2pt), "passing_yards"),
        "rush": ("rusher_player_id", not_2pt, "rushing_yards"),
        "rec": ("receiver_player_id", pc.and_(is_target, not_2pt), "receiving_yards"),
    }
    out: PlayerGames = defaultdict(list)
    for stat, (pid, mask, ycol) in specs.items():
        keep = pc.and_(pc.and_(mask, pc.is_valid(t[pid])), pc.is_valid(t["posteam"]))
        sub = pa.table({"game_id": t["game_id"], "player": t[pid], "team": t["posteam"],
                        "opp": pa.array([1.0] * len(t)), "yards": f(ycol)}).filter(keep)
        d = sub.group_by(["game_id", "player", "team"]).aggregate([("opp", "sum"), ("yards", "sum")]).to_pydict()
        for gid, p, team, n, y in zip(d["game_id"], d["player"], d["team"], d["opp_sum"], d["yards_sum"]):
            out[gid].append((p, FRANCHISE.get(team, team), stat, float(n), float(y)))
    return dict(out)


def load_player_games(files: dict[int, tuple[Path, dict]], cache_dir: Path) -> PlayerGames:
    out: PlayerGames = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for season, (path, manifest) in sorted(files.items()):
        cache = cache_dir / f"player_games_{season}_{manifest['sha256'][:12]}_{CACHE_VERSION}.csv"
        if not cache.exists():
            rows = aggregate_players_season(path)
            tmp = cache.with_suffix(".partial")
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(["game_id", "player", "team", "stat", "opp", "yards"])
                for gid in sorted(rows):
                    for r in sorted(rows[gid]):
                        w.writerow([gid, r[0], r[1], r[2], repr(r[3]), repr(r[4])])
            tmp.replace(cache)
        with open(cache, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out.setdefault(r["game_id"], []).append((r["player"], r["team"], r["stat"], float(r["opp"]), float(r["yards"])))
    return out


def player_names(paths: list[Path]) -> dict[str, str]:
    """Display names (e.g. 'P.Mahomes') for player ids, from play-by-play files; later files win."""
    import pyarrow.parquet as pq

    names: dict[str, str] = {}
    for path in paths:
        t = pq.read_table(path, columns=[f"{k}_player_{c}" for k in ("passer", "rusher", "receiver")
                                          for c in ("id", "name")]).to_pydict()
        for k in ("passer", "rusher", "receiver"):
            for i, n in zip(t[f"{k}_player_id"], t[f"{k}_player_name"]):
                if i and n:
                    names[i] = n
    return names
