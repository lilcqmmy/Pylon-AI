"""Elo ratings with home-field advantage, margin-of-victory scaling and between-season regression.

Ratings follow `Game.home_key`/`away_key` (the franchise when known), so relocations keep their
rating. Home-field advantage is not applied at neutral sites.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from sfi.data.schema import Game


@dataclass(frozen=True)
class EloParams:
    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_field_elo: float = 48.0
    season_regression: float = 0.33
    margin_of_victory: bool = True

    def __post_init__(self) -> None:
        # Normalize types so YAML's 20 and 20.0 give the same fingerprint.
        for f in ("initial_rating", "k_factor", "home_field_elo", "season_regression"):
            object.__setattr__(self, f, float(getattr(self, f)))
        object.__setattr__(self, "margin_of_victory", bool(self.margin_of_victory))

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:8]


class EloModel:
    feature_spec_version = "elo_v2"

    def __init__(self, params: EloParams | None = None, name: str = "baseline_elo"):
        self.p = params or EloParams()
        self.name = name
        # Different parameters are a different model: encode them in the version.
        self.version = f"0.3.0+{self.p.fingerprint()}"
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

    def _home_edge(self, game: Game) -> float:
        hfa = 0.0 if game.neutral_site else self.p.home_field_elo
        return self.ratings[game.home_key] + hfa - self.ratings[game.away_key]

    def predict_home_win(self, game: Game) -> float:
        return 1.0 / (1.0 + 10.0 ** (-self._home_edge(game) / 400.0))

    def observe(self, game: Game) -> None:
        if self.season is None or game.season != self.season:
            raise ValueError(f"{game.game_id}: observed outside the active season {self.season}")
        expected = self.predict_home_win(game)
        outcome = game.home_outcome
        mult = 1.0
        margin = abs(game.home_score - game.away_score)
        if self.p.margin_of_victory and margin > 0:
            winner_diff = self._home_edge(game) * (1 if outcome == 1.0 else -1)
            mult = math.log(margin + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
        delta = self.p.k_factor * mult * (outcome - expected)
        self.ratings[game.home_key] += delta
        self.ratings[game.away_key] -= delta
        self._season_games[game.home_key] += 1
        self._season_games[game.away_key] += 1

    def prior_games(self, team: str) -> int:
        return self._season_games[team]

    def inputs_for(self, game: Game) -> dict[str, Any]:
        return {
            "home_elo": round(self.ratings[game.home_key], 3),
            "away_elo": round(self.ratings[game.away_key], 3),
            "neutral_site": game.neutral_site,
            "params": asdict(self.p),
            "home_prior_games_season": self.prior_games(game.home_key),
            "away_prior_games_season": self.prior_games(game.away_key),
        }
