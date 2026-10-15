"""Elo grid search scored only on validation seasons.

Games after the last validation season are dropped before anything runs, so holdout results cannot
influence the chosen parameters even indirectly.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from sfi.data.schema import Game
from sfi.evaluation import metrics as M
from sfi.evaluation.walkforward import iter_walkforward
from sfi.models.elo import EloModel, EloParams


@dataclass(frozen=True)
class Trial:
    params: EloParams
    brier: float
    log_loss: float
    n: int


def evaluate_params(games: list[Game], params: EloParams, seasons: set[int]) -> Trial:
    last = max(seasons)
    usable = [g for g in games if g.season <= last]
    p, y = [], []
    for f in iter_walkforward(usable, EloModel(params), with_inputs=False):
        if f.game.season in seasons and f.game.home_outcome != 0.5:
            p.append(f.p_home_win)
            y.append(f.game.home_outcome)
    if not p:
        raise ValueError(f"no scorable games in seasons {sorted(seasons)}")
    return Trial(params, M.brier(p, y), M.log_loss(p, y), len(p))


def grid_params(base: EloParams, grid: dict[str, list]) -> list[EloParams]:
    unknown = set(grid) - set(EloParams.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown Elo parameters in grid: {sorted(unknown)}")
    keys = sorted(grid)
    return [
        EloParams(**{**base.__dict__, **dict(zip(keys, combo))})
        for combo in itertools.product(*(grid[k] for k in keys))
    ]


def tune_elo(
    games: list[Game], grid: dict[str, list], validation_seasons: set[int],
    base: EloParams = EloParams(), objective: str = "brier",
) -> list[Trial]:
    """All trials, best first. Ties keep grid order, so results are deterministic."""
    if objective not in ("brier", "log_loss"):
        raise ValueError("objective must be 'brier' or 'log_loss'")
    trials = [evaluate_params(games, p, validation_seasons) for p in grid_params(base, grid)]
    return sorted(trials, key=lambda t: getattr(t, objective))
