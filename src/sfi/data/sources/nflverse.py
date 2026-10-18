"""nflverse schedules/results adapter.

Data: nflverse (nflverse-data, CC-BY 4.0); schedules maintained by Lee Sharpe (nfldata).
Evaluation and column classification: docs/data_sources/nflverse_schedules.md.

Snapshots are immutable folders `data/raw/nflverse/schedules/<retrieved_at_utc>/` holding the raw
`games.csv`, upstream `timestamp.json` and our `manifest.json`. Normalization reads a snapshot and
returns `Game` records (schedule, venue and outcome only) plus a separate market table that models
never see.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from sfi.data.schema import Game, ValidationError, validate_games

ADAPTER_VERSION = "nflverse_schedules_v1"
SOURCE_NAME = "nflverse_schedules"
BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/"
LICENSE = "CC-BY-4.0"
ATTRIBUTION = "Data: nflverse (nflverse-data, CC-BY 4.0); schedules maintained by Lee Sharpe (nfldata)."
MAX_BYTES = 50 * 1024 * 1024
EASTERN = ZoneInfo("America/New_York")

FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
CURRENT_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND",
    "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA",
    "SF", "TB", "TEN", "WAS",
}
KNOWN_TEAMS = CURRENT_TEAMS | set(FRANCHISE)
NA = ("", "NA")


# --------------------------------------------------------------------------- fetch

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def latest_snapshot(root: Path) -> Path | None:
    """Most recent snapshot folder that has a manifest (folders without one are ignored)."""
    if not root.exists():
        return None
    snaps = sorted(p for p in root.iterdir() if (p / "manifest.json").exists())
    return snaps[-1] if snaps else None


def register_snapshot(folder: Path, *, retrieved_at: datetime, source_url: str) -> dict:
    """Write manifest.json for a folder that already holds games.csv (+ optional timestamp.json)."""
    data = (folder / "games.csv").read_bytes()
    upstream = None
    ts = folder / "timestamp.json"
    if ts.exists():
        upstream = json.loads(ts.read_text(encoding="utf-8")).get("last_updated")
    manifest = {
        "source_name": SOURCE_NAME, "source_url": source_url, "adapter_version": ADAPTER_VERSION,
        "retrieved_at_utc": retrieved_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "upstream_last_updated": upstream, "bytes": len(data), "sha256": _sha256(data),
        "license": LICENSE, "attribution": ATTRIBUTION,
    }
    path = folder / "manifest.json"
    if path.exists():
        raise FileExistsError(f"{path} already exists; snapshots are immutable")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _http_get(url: str) -> bytes:
    with urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https URL)
        data = resp.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"{url} exceeds {MAX_BYTES} bytes")
    return data


def fetch_schedules(
    root: Path, *, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    get: Callable[[str], bytes] = _http_get,
) -> tuple[Path, dict, bool]:
    """Download a new snapshot. Returns (folder, manifest, created). Identical bytes -> not created."""
    games = get(BASE_URL + "games.csv")
    stamp = get(BASE_URL + "timestamp.json")
    latest = latest_snapshot(root)
    if latest is not None:
        m = json.loads((latest / "manifest.json").read_text(encoding="utf-8"))
        if m["sha256"] == _sha256(games):
            return latest, m, False
    retrieved = now()
    folder = root / retrieved.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tmp = folder.with_name(folder.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        (tmp / "games.csv").write_bytes(games)
        (tmp / "timestamp.json").write_bytes(stamp)
        tmp.rename(folder)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return folder, register_snapshot(folder, retrieved_at=retrieved, source_url=BASE_URL + "games.csv"), True


# ----------------------------------------------------------------------- normalize

@dataclass(frozen=True)
class Snapshot:
    games: list[Game]
    market_p_home: dict[str, float]  # de-vigged moneyline probability; benchmark only, never a feature
    manifest: dict
    repairs: list[str]  # documented corrections applied to upstream values
    lines: dict[str, dict] = None  # {game_id: {"spread": home-favored points, "total": points}}; benchmark only

    @property
    def snapshot_id(self) -> str:
        return "sha256:" + self.manifest["sha256"][:16]


def american_to_prob(odds: float) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def market_probability(home_ml: str, away_ml: str) -> float | None:
    """Home win probability from moneylines with the bookmaker margin removed proportionally."""
    if home_ml in NA or away_ml in NA:
        return None
    try:
        ph, pa = american_to_prob(float(home_ml)), american_to_prob(float(away_ml))
    except ValueError:
        return None
    return ph / (ph + pa)


def kickoff_utc(gameday: str, gametime: str) -> datetime:
    """nflverse gametime is US Eastern regardless of venue."""
    local = datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=EASTERN)
    return local.astimezone(timezone.utc)


def _score(s: str) -> int | None:
    return None if s in NA else int(s)


def repair_gametime(season: int, gametime: str) -> tuple[str, str | None]:
    """Known upstream defect (found 2026-10-15): 2000-2005 night games are stored as "09:00" instead of
    "21:00" (101 games, all Mon/Thu/Sat). Genuine morning games are international 09:30 kickoffs from
    2014 on, so the rule is limited to exactly this pattern. Returns (gametime, note or None)."""
    if season <= 2005 and gametime == "09:00":
        return "21:00", "gametime 09:00 -> 21:00 (pre-2006 night-game defect)"
    return gametime, None


def normalize_rows(
    rows: list[dict[str, str]], *, min_season: int, known_at_hours: float,
) -> tuple[list[Game], dict[str, float], list[str]]:
    policy = f"kickoff_plus_{known_at_hours:g}h"
    games, market, problems, repairs = [], {}, [], []
    for r in rows:
        season = int(r["season"])
        if season < min_season:
            continue
        gid = r["game_id"]
        for side in ("home_team", "away_team"):
            if r[side] not in KNOWN_TEAMS:
                problems.append(f"{gid}: unknown team code {r[side]!r}")
        if r["gametime"] in NA:
            problems.append(f"{gid}: missing gametime")
            continue
        gametime, note = repair_gametime(season, r["gametime"])
        if note:
            repairs.append(f"{gid}: {note}")
        start = kickoff_utc(r["gameday"], gametime)
        hs, as_ = _score(r["home_score"]), _score(r["away_score"])
        if hs is not None and as_ is not None:
            if r["result"] not in NA and int(r["result"]) != hs - as_:
                problems.append(f"{gid}: result != home_score - away_score")
            if r["total"] not in NA and int(r["total"]) != hs + as_:
                problems.append(f"{gid}: total != home_score + away_score")
        final = hs is not None and as_ is not None
        games.append(Game(
            league="NFL", season=season, week=int(r["week"]), game_id=gid,
            home_team=r["home_team"], away_team=r["away_team"], scheduled_start_utc=start,
            home_score=hs, away_score=as_,
            result_known_at_utc=start + timedelta(hours=known_at_hours) if final else None,
            source_name=SOURCE_NAME, game_type=r["game_type"], neutral_site=r["location"] == "Neutral",
            home_franchise=FRANCHISE.get(r["home_team"], r["home_team"]),
            away_franchise=FRANCHISE.get(r["away_team"], r["away_team"]),
            source_record_id=gid, known_at_policy=policy if final else None,
        ))
        p = market_probability(r.get("home_moneyline", ""), r.get("away_moneyline", ""))
        if p is not None:
            market[gid] = p
    problems += check_team_timelines(games)
    if problems:
        raise ValidationError("; ".join(problems[:10]) + (f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""))
    validate_games(games)
    return games, market, repairs


def check_team_timelines(games: list[Game]) -> list[str]:
    """Each franchise's previous result must be known before its next kickoff, and kickoffs must be
    at least 3 days apart. Catches wrong kickoff times that would otherwise leak results early."""
    by_team: dict[str, list[Game]] = {}
    for g in games:
        by_team.setdefault(g.home_key, []).append(g)
        by_team.setdefault(g.away_key, []).append(g)
    problems = []
    for team, gs in by_team.items():
        gs.sort(key=lambda g: g.scheduled_start_utc)
        for prev, nxt in zip(gs, gs[1:]):
            if nxt.scheduled_start_utc - prev.scheduled_start_utc < timedelta(days=3):
                problems.append(f"{team}: {prev.game_id} and {nxt.game_id} kick off <3 days apart")
            if prev.result_known_at_utc and prev.result_known_at_utc > nxt.scheduled_start_utc:
                problems.append(f"{team}: {prev.game_id} result known after {nxt.game_id} kickoff")
    return problems


def load_snapshot(folder: Path, *, min_season: int, known_at_hours: float) -> Snapshot:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    path = folder / "games.csv"
    if _sha256(path.read_bytes()) != manifest["sha256"]:
        raise ValidationError(f"{path} does not match its manifest checksum")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    games, market, repairs = normalize_rows(rows, min_season=min_season, known_at_hours=known_at_hours)
    return Snapshot(games, market, manifest, repairs, market_lines(rows, min_season))


def market_lines(rows: list[dict[str, str]], min_season: int) -> dict[str, dict]:
    """Closing-ish spread and total lines (recording time undocumented). Benchmark only, never a feature."""
    out = {}
    for r in rows:
        if int(r["season"]) < min_season:
            continue
        ln = {}
        for key, col in (("spread", "spread_line"), ("total", "total_line")):
            v = r.get(col, "NA")
            if v not in NA:
                try:
                    ln[key] = float(v)
                except ValueError:
                    pass
        if ln:
            out[r["game_id"]] = ln
    return out
