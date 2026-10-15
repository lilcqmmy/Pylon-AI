"""Interface every win-probability model implements.

Models only learn through `observe()`, which the walk-forward harness calls with results that
were already known at the forecast's as-of time. They never see the schedule's future outcomes.
"""

from __future__ import annotations

from typing import Any, Protocol

from sfi.data.schema import Game


class WinModel(Protocol):
    name: str
    version: str
    feature_spec_version: str

    def start_season(self, season: int) -> None: ...

    def observe(self, game: Game) -> None: ...

    def predict_home_win(self, game: Game) -> float:
        """P(home wins | game does not end in a tie)."""
        ...

    def inputs_for(self, game: Game) -> dict[str, Any]:
        """Model inputs to archive alongside the forecast."""
        ...

    def prior_games(self, team: str) -> int:
        """Games observed for this team in the current season."""
        ...
