"""Naive benchmarks every real model must beat out-of-sample."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sfi.data.schema import Game


class _SeasonCounter:
    def __init__(self) -> None:
        self._season_games: dict[str, int] = defaultdict(int)
        self.season: int | None = None

    def start_season(self, season: int) -> None:
        if self.season is not None and season != self.season:
            self._season_games.clear()
        self.season = season

    def _count(self, game: Game) -> None:
        self._season_games[game.home_team] += 1
        self._season_games[game.away_team] += 1

    def prior_games(self, team: str) -> int:
        return self._season_games[team]


class CoinFlip(_SeasonCounter):
    name = "coin_flip"
    version = "0.1.0"
    feature_spec_version = "none"

    def observe(self, game: Game) -> None:
        self._count(game)

    def predict_home_win(self, game: Game) -> float:
        return 0.5

    def inputs_for(self, game: Game) -> dict[str, Any]:
        return {}


class HomeRate(_SeasonCounter):
    """Historical home win rate among non-tied games, with a Beta(1, 1) prior."""

    name = "home_rate"
    version = "0.1.0"
    feature_spec_version = "home_rate_v1"

    def __init__(self) -> None:
        super().__init__()
        self.home_wins = 0
        self.decided = 0

    def observe(self, game: Game) -> None:
        self._count(game)
        if game.home_outcome != 0.5:
            self.home_wins += int(game.home_outcome)
            self.decided += 1

    def predict_home_win(self, game: Game) -> float:
        return (self.home_wins + 1) / (self.decided + 2)

    def inputs_for(self, game: Game) -> dict[str, Any]:
        return {"home_wins": self.home_wins, "decided_games": self.decided}
