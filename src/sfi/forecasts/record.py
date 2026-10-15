"""Immutable forecast records and an append-only JSONL archive.

A record is hashed over its canonical JSON. The archive never rewrites history: appending a
forecast_id that already exists is allowed only if the record is byte-for-byte identical
(re-running a deterministic backtest), otherwise it raises.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sfi.data.schema import Game


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class QualityGate:
    status: str  # "publish" | "abstain"
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForecastRecord:
    forecast_id: str
    league: str
    season: int
    week: int
    game_id: str
    home_team: str
    away_team: str
    outcome_type: str
    prediction_as_of_utc: str
    scheduled_start_utc: str
    source_snapshot_id: str
    feature_spec_version: str
    model_name: str
    model_version: str
    distribution: dict[str, float]
    inputs: dict[str, Any]
    quality_gate: QualityGate
    data_kind: str  # "synthetic" | "real"
    simulation_only: bool
    record_hash: str = field(default="")

    def canonical_json(self) -> str:
        d = asdict(self)
        d.pop("record_hash")
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, sort_keys=True, separators=(",", ":"))


def make_record(
    *, game: Game, as_of: datetime, p_home_win: float, model_name: str, model_version: str,
    feature_spec_version: str, source_snapshot_id: str, inputs: dict[str, Any], gate: QualityGate,
    data_kind: str,
) -> ForecastRecord:
    if as_of >= game.scheduled_start_utc:
        raise ValueError(f"{game.game_id}: forecast as_of {as_of} is not before kickoff")
    if not 0.0 < p_home_win < 1.0:
        raise ValueError(f"p_home_win must be in (0, 1), got {p_home_win}")
    if data_kind not in ("synthetic", "real"):
        raise ValueError("data_kind must be 'synthetic' or 'real'")
    stamp = as_of.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rec = ForecastRecord(
        forecast_id=f"{model_name}-{model_version}-{game.game_id}-{stamp}",
        league=game.league, season=game.season, week=game.week, game_id=game.game_id,
        home_team=game.home_team, away_team=game.away_team,
        outcome_type="home_win_excl_tie",
        prediction_as_of_utc=_iso(as_of), scheduled_start_utc=_iso(game.scheduled_start_utc),
        source_snapshot_id=source_snapshot_id, feature_spec_version=feature_spec_version,
        model_name=model_name, model_version=model_version,
        distribution={"p_home_win": round(p_home_win, 6), "p_away_win": round(1 - p_home_win, 6)},
        inputs=inputs, quality_gate=gate, data_kind=data_kind,
        simulation_only=(data_kind == "synthetic"),
    )
    digest = hashlib.sha256(rec.canonical_json().encode()).hexdigest()
    return ForecastRecord(**{**asdict(rec), "quality_gate": gate, "record_hash": digest})


class ArchiveConflict(RuntimeError):
    pass


class ForecastArchive:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._hashes: dict[str, str] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        d = json.loads(line)
                        self._hashes[d["forecast_id"]] = d["record_hash"]

    def __len__(self) -> int:
        return len(self._hashes)

    def append(self, records: list[ForecastRecord]) -> tuple[int, int]:
        """Returns (written, already_present_identical). Raises ArchiveConflict on any rewrite."""
        new, same = [], 0
        for r in records:
            existing = self._hashes.get(r.forecast_id)
            if existing is None:
                new.append(r)
            elif existing == r.record_hash:
                same += 1
            else:
                raise ArchiveConflict(f"{r.forecast_id} already archived with a different hash")
        if new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                for r in new:
                    fh.write(r.to_json() + "\n")
                    self._hashes[r.forecast_id] = r.record_hash
        return len(new), same
