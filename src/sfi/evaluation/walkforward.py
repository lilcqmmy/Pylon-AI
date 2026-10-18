"""Strictly forward-in-time backtest.

Games are grouped into slates by kickoff time. For each slate, the as-of time is `as_of_lead` before
kickoff. A model is first fed only the results whose `result_known_at_utc` is <= that as-of time,
then forecasts every game in the slate. So a Sunday-night forecast can use Sunday-afternoon results
once they are known, but no result can reach the model before it was available.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterator

from sfi.data.schema import Game
from sfi.forecasts.record import ForecastRecord, QualityGate, make_record
from sfi.models.base import WinModel


@dataclass(frozen=True)
class GateConfig:
    min_prior_games_per_team: int = 0


@dataclass(frozen=True)
class Forecast:
    game: Game
    as_of: datetime
    p_home_win: float
    gate: QualityGate
    inputs: dict[str, Any] = field(default_factory=dict)


def quality_gate(model: WinModel, game: Game, cfg: GateConfig) -> QualityGate:
    reasons = []
    for side, team in (("home", game.home_key), ("away", game.away_key)):
        n = model.prior_games(team)
        if n < cfg.min_prior_games_per_team:
            reasons.append(f"sparse_data:{side}_team_has_{n}_games_this_season")
    return QualityGate("abstain" if reasons else "publish", tuple(reasons))


class Replay:
    """Feeds a model results strictly in known-at order and issues forecasts at given as-of times.
    Shared by the backtest and by prospective (live) forecasting."""

    def __init__(self, games: list[Game], model: WinModel):
        self.model = model
        self._pending = sorted(
            (g for g in games if g.is_final and g.result_known_at_utc is not None),
            key=lambda g: (g.result_known_at_utc, g.scheduled_start_utc, g.game_id),
        )
        self._i = 0
        self.as_of: datetime | None = None

    def advance_to(self, as_of: datetime) -> None:
        if self.as_of is not None and as_of < self.as_of:
            raise ValueError("as-of time cannot move backwards")
        self.as_of = as_of
        batch = []
        while self._i < len(self._pending) and self._pending[self._i].result_known_at_utc <= as_of:
            batch.append(self._pending[self._i])
            self._i += 1
        for g in sorted(batch, key=lambda g: (g.scheduled_start_utc, g.game_id)):
            if g.season != getattr(self.model, "season", None):
                self.model.start_season(g.season)
            self.model.observe(g)
        # Models that read time-stamped pregame information (e.g. injury reports) need the clock.
        set_as_of = getattr(self.model, "set_as_of", None)
        if set_as_of is not None:
            set_as_of(as_of)

    def forecast(self, game: Game, gate: GateConfig = GateConfig(), with_inputs: bool = True) -> Forecast:
        if self.as_of is None or self.as_of >= game.scheduled_start_utc:
            raise ValueError(f"{game.game_id}: forecast must be issued before kickoff")
        if getattr(self.model, "season", None) != game.season:
            self.model.start_season(game.season)
        p = self.model.predict_home_win(game)
        return Forecast(game, self.as_of, p, quality_gate(self.model, game, gate),
                        self.model.inputs_for(game) if with_inputs else {})


def slates_by_kickoff(games: list[Game]) -> list[tuple[datetime, list[Game]]]:
    slates: dict[datetime, list[Game]] = defaultdict(list)
    for g in games:
        slates[g.scheduled_start_utc].append(g)
    return [(k, sorted(slates[k], key=lambda g: g.game_id)) for k in sorted(slates)]


def iter_walkforward(
    games: list[Game], model: WinModel, *, gate: GateConfig = GateConfig(),
    as_of_lead: timedelta = timedelta(hours=1), with_inputs: bool = True,
) -> Iterator[Forecast]:
    replay = Replay(games, model)
    for kickoff, slate in slates_by_kickoff(games):
        replay.advance_to(kickoff - as_of_lead)
        for g in slate:
            yield replay.forecast(g, gate, with_inputs)


def run_walkforward(
    games: list[Game],
    model_factories: list[Callable[[], WinModel]],
    *,
    source_snapshot_id: str,
    data_kind: str,
    gate: GateConfig = GateConfig(),
    as_of_lead: timedelta = timedelta(hours=1),
) -> list[ForecastRecord]:
    records: list[ForecastRecord] = []
    for factory in model_factories:
        model = factory()
        for f in iter_walkforward(games, model, gate=gate, as_of_lead=as_of_lead):
            records.append(make_record(
                game=f.game, as_of=f.as_of, p_home_win=f.p_home_win,
                model_name=model.name, model_version=model.version,
                feature_spec_version=model.feature_spec_version,
                source_snapshot_id=source_snapshot_id, inputs=f.inputs,
                gate=f.gate, data_kind=data_kind,
            ))
    return records
