"""Elo ratings with home-field advantage, margin-of-victory scaling and between-season regression."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sfi.data.schema import Game


@dataclass(frozen=True)
class EloParams:
    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_field_elo: float = 48.0
    season_regression: float = 0.33
    margin_of_victory: bool = True


class EloModel:
    name = "baseline_elo"
    version = "0.1.0"
    feature_spec_version = "elo_v1"

    def __init__(self, params: EloParams | None = None):
        self.p = params or EloParams()
        self.ratings: dict[str, float] = defaultdict(lambda: self.p.initial_rating)
        self._season_games: dict[str, int] = defaultdict(int)
        self.season: int | None = None

    def start_season(self, season: int) -> None:
        if self.season is not None and season < self.season:
            raise ValueError("seasons must be processed in order")
        if self.season is not None and season > self.season:
            mean = self.p.initial_rating
            for team in list(self.ratings):
                self.ratings[team] += self.p.season_regression * (mean - self.ratings[team])
            self._season_games.clear()
        self.season = season

    def _expected_home(self, home: str, away: str) -> float:
        diff = self.ratings[home] + self.p.home_field_elo - self.ratings[away]
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def predict_home_win(self, game: Game) -> float:
        return self._expected_home(game.home_team, game.away_team)

    def observe(self, game: Game) -> None:
        if self.season is None or game.season != self.season:
            raise ValueError(f"{game.game_id}: observed outside the active season {self.season}")
        expected = self._expected_home(game.home_team, game.away_team)
        outcome = game.home_outcome
        mult = 1.0
        margin = abs(game.home_score - game.away_score)
        if self.p.margin_of_victory and margin > 0:
            winner_diff = (
                self.ratings[game.home_team] + self.p.home_field_elo - self.ratings[game.away_team]
            ) * (1 if outcome == 1.0 else -1)
            mult = math.log(margin + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
        delta = self.p.k_factor * mult * (outcome - expected)
        self.ratings[game.home_team] += delta
        self.ratings[game.away_team] -= delta
        self._season_games[game.home_team] += 1
        self._season_games[game.away_team] += 1

    def prior_games(self, team: str) -> int:
        return self._season_games[team]

    def inputs_for(self, game: Game) -> dict[str, Any]:
        return {
            "home_elo": round(self.ratings[game.home_team], 3),
            "away_elo": round(self.ratings[game.away_team], 3),
            "home_field_elo": self.p.home_field_elo,
            "home_prior_games_season": self.prior_games(game.home_team),
            "away_prior_games_season": self.prior_games(game.away_team),
        }
