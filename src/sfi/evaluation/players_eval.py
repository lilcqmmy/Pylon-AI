"""Walk-forward player yardage forecasts: runner, compact records, and evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import timedelta, timezone
from typing import Iterator

import numpy as np

from sfi.data.schema import Game
from sfi.evaluation.scores_eval import TargetEval, LEVELS, PINBALL_QS
from sfi.evaluation.walkforward import Replay, slates_by_kickoff
from sfi.features.players import PlayerGames
from sfi.forecasts.record import ForecastRecord, QualityGate, _iso
from sfi.models.players import PlayerForecast


def iter_player_walkforward(games: list[Game], model, *, as_of_lead: timedelta = timedelta(hours=1)) -> Iterator[tuple]:
    """Yields (as_of, PlayerForecast) for every pool player of every game."""
    replay = Replay(games, model)
    for kickoff, slate in slates_by_kickoff(games):
        replay.advance_to(kickoff - as_of_lead)
        for g in slate:
            replay.prepare(g)
            for f in model.predict_players(g):
                yield replay.as_of, f


def make_player_record(as_of, f: PlayerForecast, model, *, source_snapshot_id: str, data_kind: str) -> ForecastRecord:
    g = f.game
    if as_of >= g.scheduled_start_utc:
        raise ValueError(f"{g.game_id}: forecast as_of is not before kickoff")
    stamp = as_of.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap = source_snapshot_id.split(":")[-1][:12]
    rec = ForecastRecord(
        forecast_id=f"{model.name}-{model.version}-{snap}-{g.game_id}-{stamp}-{f.player}-{f.stat}",
        league=g.league, season=g.season, week=g.week, game_id=g.game_id, home_team=g.home_team,
        away_team=g.away_team, outcome_type=f"player_{f.stat}_yards", prediction_as_of_utc=_iso(as_of),
        scheduled_start_utc=_iso(g.scheduled_start_utc), source_snapshot_id=source_snapshot_id,
        feature_spec_version=model.feature_spec_version, model_name=model.name, model_version=model.version,
        distribution={"player": f.player, "team": f.team, "mu": round(f.mu, 3),
                      **f.dist.summary((0.1, 0.25, 0.5, 0.75, 0.9))},
        inputs=f.inputs, quality_gate=QualityGate("publish"), data_kind=data_kind,
        simulation_only=data_kind == "synthetic",
    )
    digest = hashlib.sha256(rec.canonical_json().encode()).hexdigest()
    return ForecastRecord(**{**asdict(rec), "quality_gate": rec.quality_gate, "record_hash": digest})


def outcomes(stats: PlayerGames) -> dict[tuple[str, str, str], tuple[float, float]]:
    return {(gid, p, s): (n, y) for gid, rows in stats.items() for p, _, s, n, y in rows}


def evaluate_players(forecasts: list[tuple[str, object, PlayerForecast]], stats: PlayerGames, seasons: set[int],
                     seed: int = 0) -> tuple[dict[tuple[str, str], TargetEval], dict[tuple[str, str], dict]]:
    """Returns ({(model, stat): TargetEval}, {(model, stat): extra counts}). Pool players with no
    opportunity in the game score as 0 yards (counted as `no_opportunity`)."""
    rng = np.random.default_rng(seed)
    out_map = outcomes(stats)
    evals: dict[tuple[str, str], TargetEval] = {}
    extra: dict[tuple[str, str], dict] = {}
    for model, _, f in forecasts:
        g = f.game
        if g.season not in seasons or not g.is_final:
            continue
        key = (model, f.stat)
        e = evals.setdefault(key, TargetEval(f"{model}:{f.stat}"))
        x = extra.setdefault(key, {"no_opportunity": 0})
        n, y = out_map.get((g.game_id, f.player, f.stat), (0.0, 0.0))
        if n == 0:
            x["no_opportunity"] += 1
        y = int(round(y))
        d = f.dist
        e.game_ids.append(f"{g.game_id}|{f.player}|{f.stat}")
        e.blocks.append(f"{g.season}-W{g.week:02d}")
        e.y.append(y)
        e.crps.append(d.crps(y))
        e.abs_err_median.append(abs(y - d.quantile(0.5)))
        e.sq_err_mean.append((y - d.mean()) ** 2)
        e.pinball.append(d.pinball(y, PINBALL_QS))
        for lv in LEVELS:
            a, b = d.central_interval(lv)
            e.hits[lv].append(float(a <= y <= b))
            e.claimed[lv].append(d.mass(a, b))
            e.width[lv].append(b - a)
        e.pit.append(d.pit(y, float(rng.uniform())))
    return evals, extra
