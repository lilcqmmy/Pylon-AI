"""Canonical game records, validation, CSV storage and point-in-time (as-of) access."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Game:
    league: str
    season: int
    week: int
    game_id: str
    home_team: str
    away_team: str
    scheduled_start_utc: datetime
    home_score: int | None
    away_score: int | None
    result_known_at_utc: datetime | None  # when the final result became available to us
    source_name: str

    @property
    def is_final(self) -> bool:
        return self.home_score is not None and self.away_score is not None

    @property
    def home_outcome(self) -> float:
        """1.0 home win, 0.0 away win, 0.5 tie."""
        if not self.is_final:
            raise ValueError(f"{self.game_id} has no final result")
        if self.home_score > self.away_score:
            return 1.0
        if self.home_score < self.away_score:
            return 0.0
        return 0.5

    def result_known_by(self, as_of: datetime) -> bool:
        return self.is_final and self.result_known_at_utc is not None and self.result_known_at_utc <= as_of


def known_results(games: Iterable[Game], as_of: datetime) -> list[Game]:
    """The only sanctioned way to read outcomes: results whose availability time is <= as_of."""
    return sorted(
        (g for g in games if g.result_known_by(as_of)),
        key=lambda g: (g.scheduled_start_utc, g.game_id),
    )


class ValidationError(ValueError):
    pass


def validate_games(games: list[Game]) -> None:
    problems: list[str] = []
    dup_ids = [gid for gid, n in Counter(g.game_id for g in games).items() if n > 1]
    if dup_ids:
        problems.append(f"duplicate game_id: {dup_ids[:5]}")
    slots = Counter()
    for g in games:
        if g.home_team == g.away_team:
            problems.append(f"{g.game_id}: team plays itself")
        if g.scheduled_start_utc.tzinfo is None:
            problems.append(f"{g.game_id}: naive scheduled_start_utc")
        if (g.home_score is None) != (g.away_score is None):
            problems.append(f"{g.game_id}: partial score")
        if g.is_final:
            if g.home_score < 0 or g.away_score < 0:
                problems.append(f"{g.game_id}: negative score")
            if g.result_known_at_utc is None:
                problems.append(f"{g.game_id}: final score without result_known_at_utc")
            elif g.result_known_at_utc <= g.scheduled_start_utc:
                problems.append(f"{g.game_id}: result known before kickoff")
        slots[(g.season, g.week, g.home_team)] += 1
        slots[(g.season, g.week, g.away_team)] += 1
    double_booked = [k for k, n in slots.items() if n > 1]
    if double_booked:
        problems.append(f"team scheduled twice in a week: {double_booked[:5]}")
    if problems:
        raise ValidationError("; ".join(problems))


_FIELDS = [f.name for f in fields(Game)]


def _fmt_dt(dt: datetime | None) -> str:
    return "" if dt is None else dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_dt(s: str) -> datetime | None:
    return None if s == "" else datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _parse_int(s: str) -> int | None:
    return None if s == "" else int(s)


def write_games_csv(games: list[Game], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(_FIELDS)
        for g in games:
            w.writerow([
                g.league, g.season, g.week, g.game_id, g.home_team, g.away_team,
                _fmt_dt(g.scheduled_start_utc),
                "" if g.home_score is None else g.home_score,
                "" if g.away_score is None else g.away_score,
                _fmt_dt(g.result_known_at_utc), g.source_name,
            ])


def read_games_csv(path: str | Path) -> list[Game]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    games = [
        Game(
            league=r["league"], season=int(r["season"]), week=int(r["week"]), game_id=r["game_id"],
            home_team=r["home_team"], away_team=r["away_team"],
            scheduled_start_utc=_parse_dt(r["scheduled_start_utc"]),
            home_score=_parse_int(r["home_score"]), away_score=_parse_int(r["away_score"]),
            result_known_at_utc=_parse_dt(r["result_known_at_utc"]), source_name=r["source_name"],
        )
        for r in rows
    ]
    validate_games(games)
    return games


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
