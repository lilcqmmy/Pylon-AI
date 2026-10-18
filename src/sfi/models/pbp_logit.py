"""Elo + play-by-play team features in an L2-regularized logistic regression, fit walk-forward.

Features per game: the Elo logit (home-field included) and, for each team-feature, home minus away.
At each forecast the feature vector is computed from `TeamHistory` (past games only) and remembered;
when the game's result is later observed, (vector, outcome) joins the training set. The regression is
refit at the start of each season on all past pairs. Until enough pairs exist it returns the Elo
probability unchanged.

`stats` maps (game_id, franchise) to that team's offensive sums for the game. The model reads an
entry only inside `observe()`, i.e. after the replay has released that game's result.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from sfi.data.schema import Game
from sfi.features.pbp import TeamHistory
from sfi.models.elo import EloModel, EloParams


def fit_logistic_l2(X: np.ndarray, y: np.ndarray, l2: float, iters: int = 100) -> np.ndarray:
    """IRLS for logistic regression with intercept (column 0, unpenalized) and ridge penalty l2."""
    beta = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], l2)
    pen[0] = 0.0
    for _ in range(iters):
        z = np.clip(X @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-z))
        w = np.maximum(mu * (1 - mu), 1e-9)
        H = X.T @ (X * w[:, None]) + np.diag(pen)
        g = X.T @ (y - mu) - pen * beta
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


class PbpLogit:
    feature_spec_version = "pbp_team_v1"

    def __init__(
        self, elo_params: EloParams, stats: dict[tuple[str, str], dict[str, float]], *,
        metrics: list[str] | None = None, l2: float = 10.0, min_samples: int = 200, name: str = "pbp_logit",
        data_tag: str = "",
    ):
        """`data_tag` identifies the play-by-play snapshot; it is part of the version, so forecasts made
        from different play-by-play data never share a forecast ID."""
        self.elo = EloModel(elo_params, name="elo_inner")
        self.history = TeamHistory(metrics)
        self._stats = stats
        self.l2, self.min_samples, self.name = l2, min_samples, name
        self.names = ["elo_logit"] + [f"d_{n}" for n in self.history.feature_names()]
        spec = json.dumps({"elo": self.elo.version, "metrics": self.history.metrics, "l2": l2,
                           "min": min_samples, "data": data_tag}, sort_keys=True)
        self.version = f"0.1.0+{hashlib.sha256(spec.encode()).hexdigest()[:8]}"
        self._issued: dict[str, np.ndarray] = {}
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._coef: np.ndarray | None = None
        self._mean = self._std = None
        self.fit_season: int | None = None
        self.n_missing_stats = 0

    @property
    def season(self) -> int | None:
        return self.elo.season

    def start_season(self, season: int) -> None:
        new = self.elo.season is not None and season > self.elo.season
        self.elo.start_season(season)
        if new:
            self.history.start_season()
        if len(self._X) >= self.min_samples and self.fit_season != season:
            self._refit()
            self.fit_season = season

    def _refit(self) -> None:
        X = np.vstack(self._X)
        y = np.asarray(self._y)
        mean = np.nanmean(X, axis=0)
        mean = np.where(np.isnan(mean), 0.0, mean)
        Xf = np.where(np.isnan(X), mean, X)
        std = Xf.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        Z = np.column_stack([np.ones(len(y)), (Xf - mean) / std])
        self._coef, self._mean, self._std = fit_logistic_l2(Z, y, self.l2), mean, std

    def _vector(self, game: Game) -> np.ndarray:
        p_elo = self.elo.predict_home_win(game)
        h, a = self.history.features(game.home_key), self.history.features(game.away_key)
        diffs = [h[n] - a[n] for n in self.history.feature_names()]
        return np.array([math.log(p_elo / (1 - p_elo)), *diffs], dtype=float)

    def predict_home_win(self, game: Game) -> float:
        x = self._vector(game)
        self._issued[game.game_id] = x
        if self._coef is None:
            return self.elo.predict_home_win(game)
        xf = np.where(np.isnan(x), self._mean, x)
        z = self._coef[0] + ((xf - self._mean) / self._std) @ self._coef[1:]
        return float(min(max(1 / (1 + math.exp(-z)), 1e-6), 1 - 1e-6))

    def observe(self, game: Game) -> None:
        x = self._issued.pop(game.game_id, None)
        if x is not None and game.home_outcome != 0.5:
            self._X.append(x)
            self._y.append(game.home_outcome)
        home = self._stats.get((game.game_id, game.home_key))
        away = self._stats.get((game.game_id, game.away_key))
        if home is not None and away is not None:
            self.history.observe_game(game.home_key, home, away)
            self.history.observe_game(game.away_key, away, home)
        else:
            self.n_missing_stats += 1
        self.elo.observe(game)

    def prior_games(self, team: str) -> int:
        return self.elo.prior_games(team)

    def inputs_for(self, game: Game) -> dict[str, Any]:
        x = self._issued.get(game.game_id)
        feats = {} if x is None else {n: (None if math.isnan(v) else round(float(v), 5)) for n, v in zip(self.names, x)}
        return {"p_elo": round(self.elo.predict_home_win(game), 6), "n_train": len(self._X),
                "fit_season": self.fit_season, "l2": self.l2, "features": feats}
