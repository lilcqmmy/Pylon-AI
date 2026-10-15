"""Walk-forward Platt recalibration wrapper.

The wrapper remembers the raw probability it actually issued for each game. When that game's result
is later observed (i.e. it was known by an as-of time), the (raw forecast, outcome) pair joins the
calibration sample. The mapping p' = sigmoid(a + b * logit(p)) is refit lazily before the next
forecast, so it only ever uses past issued forecasts and their known outcomes.
"""

from __future__ import annotations

import math
from typing import Any

from sfi.data.schema import Game
from sfi.evaluation.metrics import calibration_intercept_slope
from sfi.models.base import WinModel

P_CLIP = 1e-6


class PlattCalibrated:
    def __init__(
        self, base: WinModel, *, min_samples: int = 200, exclude_first_season: bool = True,
        name: str | None = None,
    ):
        self.base = base
        self.min_samples = min_samples
        self.exclude_first_season = exclude_first_season
        self.name = name or f"{base.name}_platt"
        self.version = f"{base.version}+platt-m{min_samples}{'-x1' if exclude_first_season else ''}"
        self.feature_spec_version = base.feature_spec_version
        self._issued: dict[str, float] = {}
        self._x: list[float] = []
        self._y: list[float] = []
        self._coef = (0.0, 1.0)
        self._dirty = False
        self._first_season: int | None = None

    @property
    def season(self) -> int | None:
        return getattr(self.base, "season", None)

    @property
    def coefficients(self) -> tuple[float, float]:
        return self._coef

    @property
    def n_samples(self) -> int:
        return len(self._x)

    def start_season(self, season: int) -> None:
        if self._first_season is None:
            self._first_season = season
        self.base.start_season(season)

    def observe(self, game: Game) -> None:
        raw = self._issued.pop(game.game_id, None)
        skip = self.exclude_first_season and game.season == self._first_season
        if raw is not None and game.home_outcome != 0.5 and not skip:
            self._x.append(raw)
            self._y.append(game.home_outcome)
            self._dirty = True
        self.base.observe(game)

    def _refit(self) -> None:
        self._dirty = False
        if len(self._x) < self.min_samples:
            return
        a, b = calibration_intercept_slope(self._x, self._y)
        if not (math.isnan(a) or math.isnan(b)):
            self._coef = (a, b)

    def transform(self, p: float) -> float:
        a, b = self._coef
        p = min(max(p, P_CLIP), 1 - P_CLIP)
        z = a + b * math.log(p / (1 - p))
        return min(max(1 / (1 + math.exp(-z)), P_CLIP), 1 - P_CLIP)

    def predict_home_win(self, game: Game) -> float:
        if self._dirty:
            self._refit()
        raw = self.base.predict_home_win(game)
        self._issued[game.game_id] = raw
        return self.transform(raw)

    def prior_games(self, team: str) -> int:
        return self.base.prior_games(team)

    def inputs_for(self, game: Game) -> dict[str, Any]:
        a, b = self._coef
        return {
            **self.base.inputs_for(game),
            "raw_p_home_win": round(self._issued.get(game.game_id, float("nan")), 6),
            "platt_a": round(a, 6), "platt_b": round(b, 6), "platt_n": len(self._x),
        }
