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
    game_type: str = "REG"  # REG, or a playoff round (WC, DIV, CON, SB)
    neutral_site: bool = False
    home_franchise: str | None = None  # stable across relocations (e.g. OAK and LV -> LV)
    away_franchise: str | None = None
    source_record_id: str | None = None
    known_at_policy: str | None = None  # how result_known_at_utc was derived

    @property
    def home_key(self) -> str:
        """Identity a rating should follow: the franchise if known, else the team code."""
        return self.home_franchise or self.home_team

    @property
    def away_key(self) -> str:
        return self.away_franchise or self.away_team

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
        if g.home_team == g.away_team or g.home_key == g.away_key:
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


def _parse_opt_str(s: str) -> str | None:
    return None if s == "" else s


_DT_FIELDS = {"scheduled_start_utc", "result_known_at_utc"}
_PARSERS = {
    "season": int, "week": int, "home_score": _parse_int, "away_score": _parse_int,
    "scheduled_start_utc": _parse_dt, "result_known_at_utc": _parse_dt,
    "neutral_site": lambda s: s == "1",
    "home_franchise": _parse_opt_str, "away_franchise": _parse_opt_str,
    "source_record_id": _parse_opt_str, "known_at_policy": _parse_opt_str,
}


def _fmt(name: str, value) -> str:
    if name in _DT_FIELDS:
        return _fmt_dt(value)
    if isinstance(value, bool):
        return "1" if value else "0"
    return "" if value is None else str(value)


def write_games_csv(games: list[Game], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(_FIELDS)
        for g in games:
            w.writerow([_fmt(f, getattr(g, f)) for f in _FIELDS])


def read_games_csv(path: str | Path) -> list[Game]:
    """Columns added after a file was written fall back to the Game defaults."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    games = [
        Game(**{k: _PARSERS.get(k, str)(v) for k, v in r.items() if k in _FIELDS})
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
