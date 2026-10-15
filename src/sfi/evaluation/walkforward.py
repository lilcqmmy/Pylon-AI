"""Strictly forward-in-time backtest.

For each (season, week), the as-of time is one hour before that week's earliest kickoff. Every model
is first fed only the results that were *known* by then (via `known_results`), then forecasts every
game in the week. Results from the week itself or later can never reach the model before it forecasts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from sfi.data.schema import Game, known_results
from sfi.forecasts.record import ForecastRecord, QualityGate, make_record
from sfi.models.base import WinModel


@dataclass(frozen=True)
class GateConfig:
    min_prior_games_per_team: int = 0


def quality_gate(model: WinModel, game: Game, cfg: GateConfig) -> QualityGate:
    reasons = []
    for side, team in (("home", game.home_team), ("away", game.away_team)):
        n = model.prior_games(team)
        if n < cfg.min_prior_games_per_team:
            reasons.append(f"sparse_data:{side}_team_has_{n}_games_this_season")
    return QualityGate("abstain" if reasons else "publish", tuple(reasons))


def run_walkforward(
    games: list[Game],
    model_factories: list[Callable[[], WinModel]],
    *,
    source_snapshot_id: str,
    data_kind: str,
    gate: GateConfig = GateConfig(),
    as_of_lead: timedelta = timedelta(hours=1),
) -> list[ForecastRecord]:
    weeks: dict[tuple[int, int], list[Game]] = defaultdict(list)
    for g in games:
        weeks[(g.season, g.week)].append(g)

    models = [f() for f in model_factories]
    observed: list[set[str]] = [set() for _ in models]
    records: list[ForecastRecord] = []

    for (season, week) in sorted(weeks):
        slate = sorted(weeks[(season, week)], key=lambda g: (g.scheduled_start_utc, g.game_id))
        as_of: datetime = slate[0].scheduled_start_utc - as_of_lead
        available = known_results(games, as_of)
        for model, seen in zip(models, observed):
            for g in available:
                if g.game_id in seen:
                    continue
                if g.season != getattr(model, "season", None):
                    model.start_season(g.season)
                model.observe(g)
                seen.add(g.game_id)
            if getattr(model, "season", None) != season:
                model.start_season(season)
            for g in slate:
                records.append(make_record(
                    game=g, as_of=as_of, p_home_win=model.predict_home_win(g),
                    model_name=model.name, model_version=model.version,
                    feature_spec_version=model.feature_spec_version,
                    source_snapshot_id=source_snapshot_id, inputs=model.inputs_for(g),
                    gate=quality_gate(model, g, gate), data_kind=data_kind,
                ))
    return records
