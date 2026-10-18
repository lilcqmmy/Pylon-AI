"""Elo + QB availability, fit walk-forward (see docs/data_sources/nflverse_availability.md).

Per team at forecast time:
- previous starter: the team's starter in its last released game;
- status: that QB's status on the latest injury report for this game known by the as-of time;
- expected starter: the previous starter, unless listed Out/Doubtful, in which case the team's other
  QB with the most dropbacks in released games this season (else last season; else unknown);
- QB rating: cumulative `qb_epa` per dropback from released games, shrunk toward `prior_mean` with
  `k_prior` dropbacks.

Features: Elo logit, (expected - previous starter rating) home minus away, and Out/Doubtful and
Questionable flags per side. Ridge logistic refit at each season start on past forecasts only.

Point-in-time guarantees: QB stats and starters are read only inside `observe()` (released games);
injury rows are used only if their known-at time is <= the replay clock (`set_as_of`).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from sfi.data.schema import Game
from sfi.features.qb import InjuryRow
from sfi.models.elo import EloModel, EloParams
from sfi.models.pbp_logit import fit_logistic_l2

FEATURES = ["elo_logit", "qb_delta_diff", "home_out", "away_out", "home_q", "away_q"]
CONTINUOUS = (0, 1)  # indices of FEATURES that are standardized before fitting


class QbAdjusted:
    feature_spec_version = "qb_avail_v1"

    def __init__(
        self, elo_params: EloParams, qb_games: dict[str, list[tuple[str, str, float, float]]],
        starters: dict[tuple[str, str], str], injuries: dict[tuple[int, int, str], list[InjuryRow]], *,
        k_prior: float = 300.0, prior_mean: float = 0.0, l2: float = 10.0, min_samples: int = 200,
        name: str = "elo_qb", data_tag: str = "",
    ):
        self.elo = EloModel(elo_params, name="elo_inner")
        self._qb_games, self._starters, self._injuries = qb_games, starters, injuries
        self.k_prior, self.prior_mean, self.l2, self.min_samples, self.name = k_prior, prior_mean, l2, min_samples, name
        spec = json.dumps({"elo": self.elo.version, "k": k_prior, "m": prior_mean, "l2": l2,
                           "min": min_samples, "data": data_tag}, sort_keys=True)
        self.version = f"0.2.0+{hashlib.sha256(spec.encode()).hexdigest()[:8]}"  # 0.2: raw (unscaled) flags
        self.qb_epa: dict[str, float] = defaultdict(float)
        self.qb_db: dict[str, float] = defaultdict(float)
        self.last_starter: dict[str, str] = {}
        self.team_qb_db: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.prev_team_qb_db: dict[str, dict[str, float]] = {}
        self.as_of: datetime | None = None
        self._issued: dict[str, tuple[np.ndarray, dict]] = {}
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._coef = self._mean = self._std = None
        self.fit_season: int | None = None

    # ---- replay interface
    @property
    def season(self) -> int | None:
        return self.elo.season

    def set_as_of(self, as_of: datetime) -> None:
        self.as_of = as_of

    def start_season(self, season: int) -> None:
        new = self.elo.season is not None and season > self.elo.season
        self.elo.start_season(season)
        if new:
            self.prev_team_qb_db = {t: dict(d) for t, d in self.team_qb_db.items()}
            self.team_qb_db = defaultdict(lambda: defaultdict(float))
        if len(self._X) >= self.min_samples and self.fit_season != season:
            self._refit()
            self.fit_season = season

    def observe(self, game: Game) -> None:
        issued = self._issued.pop(game.game_id, None)
        if issued is not None and game.home_outcome != 0.5:
            self._X.append(issued[0])
            self._y.append(game.home_outcome)
        for qb, team, e, n in self._qb_games.get(game.game_id, ()):
            self.qb_epa[qb] += e
            self.qb_db[qb] += n
            self.team_qb_db[team][qb] += n
        for team in (game.home_key, game.away_key):
            s = self._starters.get((game.game_id, team))
            if s is not None:
                self.last_starter[team] = s
        self.elo.observe(game)

    def prior_games(self, team: str) -> int:
        return self.elo.prior_games(team)

    # ---- features
    def rating(self, qb: str | None) -> float:
        if qb is None:
            return self.prior_mean
        return (self.qb_epa[qb] + self.k_prior * self.prior_mean) / (self.qb_db[qb] + self.k_prior)

    def _status(self, game: Game, team: str, qb: str | None) -> str | None:
        if qb is None or self.as_of is None:
            return None
        known = [r for r in self._injuries.get((game.season, game.week, team), ())
                 if r.player == qb and r.known_at <= self.as_of]
        return max(known, key=lambda r: r.known_at).status if known else None

    def _side(self, game: Game, team: str) -> dict:
        prev = self.last_starter.get(team)
        status = self._status(game, team, prev)
        out = status in ("Out", "Doubtful")
        expected = prev
        if out:
            pool = self.team_qb_db.get(team) or self.prev_team_qb_db.get(team) or {}
            cands = sorted(((n, q) for q, n in pool.items() if q != prev), reverse=True)
            expected = cands[0][1] if cands else None
        delta = self.rating(expected) - self.rating(prev) if (prev is not None and expected != prev) else 0.0
        return {"prev": prev, "status": status, "expected": expected, "delta": delta,
                "out": float(out), "q": float(status == "Questionable")}

    def _vector(self, game: Game) -> tuple[np.ndarray, dict]:
        p_elo = self.elo.predict_home_win(game)
        h, a = self._side(game, game.home_key), self._side(game, game.away_key)
        x = np.array([math.log(p_elo / (1 - p_elo)), h["delta"] - a["delta"], h["out"], a["out"], h["q"], a["q"]])
        return x, {"home": h, "away": a, "p_elo": p_elo}

    def _refit(self) -> None:
        X, y = np.vstack(self._X), np.asarray(self._y)
        # Standardize only continuous columns. Rare 0/1 flags stay raw: dividing by a tiny standard
        # deviation would turn one flagged game into a z-value of ~10+ and an absurd adjustment.
        mean, std = np.zeros(X.shape[1]), np.ones(X.shape[1])
        cont = list(CONTINUOUS)
        mean[cont] = X[:, cont].mean(axis=0)
        s = X[:, cont].std(axis=0)
        std[cont] = np.where(s > 1e-12, s, 1.0)
        Z = np.column_stack([np.ones(len(y)), (X - mean) / std])
        self._coef, self._mean, self._std = fit_logistic_l2(Z, y, self.l2), mean, std

    def predict_home_win(self, game: Game) -> float:
        x, info = self._vector(game)
        self._issued[game.game_id] = (x, info)
        if self._coef is None:
            return info["p_elo"]
        z = self._coef[0] + ((x - self._mean) / self._std) @ self._coef[1:]
        return float(min(max(1 / (1 + math.exp(-z)), 1e-6), 1 - 1e-6))

    def inputs_for(self, game: Game) -> dict[str, Any]:
        x, info = self._issued.get(game.game_id, (None, None))
        if info is None:
            return {}

        def side(s):
            return {"prev_starter": s["prev"], "status": s["status"], "expected_starter": s["expected"],
                    "rating_delta": round(s["delta"], 5)}

        return {"p_elo": round(info["p_elo"], 6), "home": side(info["home"]), "away": side(info["away"]),
                "n_train": len(self._X), "fit_season": self.fit_season,
                "as_of": self.as_of.strftime("%Y-%m-%dT%H:%M:%SZ") if self.as_of else None}
