"""`elo_sched`: tuned Elo + schedule context in a ridge logistic (L2 fixed), refit each season start on past
forecasts only. Continuous features (Elo logit, rest difference, travel difference, time-zone difference)
are standardized; 0/1 and -1/0/1 flags stay raw."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from sfi.data.schema import Game
from sfi.features.schedule import CONTINUOUS, FEATURES, feature_vector
from sfi.models.elo import EloModel, EloParams
from sfi.models.pbp_logit import fit_logistic_l2

NAMES = ["elo_logit", *FEATURES]


class EloSched:
    feature_spec_version = "elo_sched_v1"

    def __init__(self, elo_params: EloParams, context: dict[str, dict], *, l2: float = 10.0, min_samples: int = 200,
                 name: str = "elo_sched", data_tag: str = ""):
        self.elo = EloModel(elo_params, name="elo_inner")
        self.context, self.l2, self.min_samples, self.name = context, l2, min_samples, name
        spec = json.dumps({"elo": self.elo.version, "l2": l2, "min": min_samples, "f": FEATURES, "data": data_tag},
                          sort_keys=True)
        self.version = f"0.1.0+{hashlib.sha256(spec.encode()).hexdigest()[:8]}"
        self._cont = [0] + [i + 1 for i, f in enumerate(FEATURES) if f in CONTINUOUS]
        self._issued: dict[str, np.ndarray] = {}
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._coef = self._mean = self._std = None
        self.fit_season: int | None = None

    @property
    def season(self) -> int | None:
        return self.elo.season

    def start_season(self, season: int) -> None:
        self.elo.start_season(season)
        if len(self._X) >= self.min_samples and self.fit_season != season:
            X, y = np.vstack(self._X), np.asarray(self._y)
            mean, std = np.zeros(X.shape[1]), np.ones(X.shape[1])
            mean[self._cont] = X[:, self._cont].mean(axis=0)
            s = X[:, self._cont].std(axis=0)
            std[self._cont] = np.where(s > 1e-12, s, 1.0)
            Z = np.column_stack([np.ones(len(y)), (X - mean) / std])
            self._coef, self._mean, self._std = fit_logistic_l2(Z, y, self.l2), mean, std
            self.fit_season = season

    def observe(self, g: Game) -> None:
        x = self._issued.pop(g.game_id, None)
        if x is not None and g.home_outcome != 0.5:
            self._X.append(x)
            self._y.append(g.home_outcome)
        self.elo.observe(g)

    def prior_games(self, team: str) -> int:
        return self.elo.prior_games(team)

    def predict_home_win(self, g: Game) -> float:
        p = self.elo.predict_home_win(g)
        ctx = self.context.get(g.game_id)
        feats = feature_vector(ctx) if ctx else [0.0] * len(FEATURES)
        x = np.array([math.log(p / (1 - p)), *feats])
        self._issued[g.game_id] = x
        if self._coef is None:
            return p
        z = self._coef[0] + ((x - self._mean) / self._std) @ self._coef[1:]
        return float(min(max(1 / (1 + math.exp(-z)), 1e-6), 1 - 1e-6))

    def inputs_for(self, g: Game) -> dict[str, Any]:
        x = self._issued.get(g.game_id)
        ctx = self.context.get(g.game_id, {})
        return {"features": {} if x is None else {n: round(float(v), 4) for n, v in zip(NAMES, x)},
                "away_travel_mi": ctx.get("away_travel_mi"), "home_travel_mi": ctx.get("home_travel_mi"),
                "coef": None if self._coef is None else [round(float(c), 5) for c in self._coef],
                "fit_season": self.fit_season}
