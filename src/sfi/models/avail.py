"""`elo_avail`: Elo + weighted starter availability, ridge logistic refit walk-forward.

Per team at forecast time: expected starters (depth chart, see `features/availability.py`) listed Out or
Doubtful on an injury report known by the replay clock, counted by position group and weighted by a fixed
importance prior. Features: Elo logit (standardized) and home-minus-away weighted availability score
(left unscaled: it is zero for most games, and standardizing rare values blows them up). Refit at each
season start on past forecasts only; Elo alone until `min_samples` pairs exist.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Any

import numpy as np

from sfi.data.schema import Game
from sfi.features.availability import GROUPS, WEIGHTS, DepthCharts
from sfi.features.qb import InjuryRow
from sfi.models.elo import EloModel, EloParams
from sfi.models.pbp_logit import fit_logistic_l2


class EloAvail:
    feature_spec_version = "elo_avail_v1"

    def __init__(self, elo_params: EloParams, depth: DepthCharts,
                 injuries: dict[tuple[int, int, str], list[InjuryRow]], *, weights: dict[str, float] | None = None,
                 l2: float = 10.0, min_samples: int = 200, name: str = "elo_avail", data_tag: str = ""):
        self.elo = EloModel(elo_params, name="elo_inner")
        self.depth, self._injuries = depth, injuries
        self.weights = dict(weights or WEIGHTS)
        self.l2, self.min_samples, self.name = l2, min_samples, name
        spec = json.dumps({"elo": self.elo.version, "w": self.weights, "l2": l2, "min": min_samples,
                           "data": data_tag}, sort_keys=True)
        self.version = f"0.1.0+{hashlib.sha256(spec.encode()).hexdigest()[:8]}"
        self.as_of: datetime | None = None
        self._issued: dict[str, tuple[np.ndarray, dict]] = {}
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._coef = self._mean = self._std = None
        self.fit_season: int | None = None

    @property
    def season(self) -> int | None:
        return self.elo.season

    def set_as_of(self, as_of: datetime) -> None:
        self.as_of = as_of

    def start_season(self, season: int) -> None:
        self.elo.start_season(season)
        if len(self._X) >= self.min_samples and self.fit_season != season:
            X, y = np.vstack(self._X), np.asarray(self._y)
            mean, std = np.zeros(X.shape[1]), np.ones(X.shape[1])
            mean[0], s = X[:, 0].mean(), X[:, 0].std()
            std[0] = s if s > 1e-12 else 1.0
            Z = np.column_stack([np.ones(len(y)), (X - mean) / std])
            self._coef, self._mean, self._std = fit_logistic_l2(Z, y, self.l2), mean, std
            self.fit_season = season

    def observe(self, g: Game) -> None:
        issued = self._issued.pop(g.game_id, None)
        if issued is not None and g.home_outcome != 0.5:
            self._X.append(issued[0])
            self._y.append(g.home_outcome)
        self.elo.observe(g)

    def prior_games(self, team: str) -> int:
        return self.elo.prior_games(team)

    def side(self, g: Game, team: str) -> dict:
        starters = self.depth.starters(g.season, g.week, team, self.as_of)
        counts = {grp: 0 for grp in GROUPS}
        if starters and self.as_of is not None:
            out = {r.player for r in self._injuries.get((g.season, g.week, team), ())
                   if r.known_at <= self.as_of and r.status in ("Out", "Doubtful")}
            for pid, grp in starters.items():
                if pid in out:
                    counts[grp] += 1
        score = sum(self.weights[grp] * n for grp, n in counts.items())
        return {"starters_known": starters is not None, "out": counts, "n_out": sum(counts.values()), "score": score}

    def predict_home_win(self, g: Game) -> float:
        p_elo = self.elo.predict_home_win(g)
        h, a = self.side(g, g.home_key), self.side(g, g.away_key)
        x = np.array([math.log(p_elo / (1 - p_elo)), h["score"] - a["score"]])
        self._issued[g.game_id] = (x, {"home": h, "away": a, "p_elo": p_elo})
        if self._coef is None:
            return p_elo
        z = self._coef[0] + ((x - self._mean) / self._std) @ self._coef[1:]
        return float(min(max(1 / (1 + math.exp(-z)), 1e-6), 1 - 1e-6))

    def inputs_for(self, g: Game) -> dict[str, Any]:
        x, info = self._issued.get(g.game_id, (None, None))
        if info is None:
            return {}
        return {"p_elo": round(info["p_elo"], 6), "home": info["home"], "away": info["away"],
                "coef": None if self._coef is None else [round(float(c), 5) for c in self._coef],
                "fit_season": self.fit_season, "n_train": len(self._X)}
