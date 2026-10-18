"""Score-margin and game-total predictive distributions, learned walk-forward.

Every model implements `predict_scores(game) -> (margin_dist, total_dist)` (margin = home - away) plus
the replay interface (`start_season`, `observe`, `prior_games`). As with the win models, a model learns
only from games the replay has released; residual spreads come from the model's own past issued
forecasts compared with outcomes known later.

- `NaiveScores`: distribution of past margins (home perspective) and recent totals. No game information.
- `EloScores`: margin mean = OLS on the Elo points edge (refit each season); total = recent league mean.
- `PointsModel`: team offensive/defensive scoring ratings, league scoring level and home edge, updated
  after each game; gives both margin and total means.

Spread: `residual_method="normal"` (discretized Normal with sd of past residuals) or `"empirical"`
(past residual shapes, which keep mass on key numbers relative to the mean).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass

import numpy as np

from sfi.data.schema import Game
from sfi.distributions import DiscreteDist, from_normal, from_residuals
from sfi.models.elo import EloModel, EloParams

MARGIN_RANGE = (-70, 70)
TOTAL_RANGE = (0, 120)


def _version(base: str, spec: dict) -> str:
    return f"{base}+{hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]}"


class _ResidualSpread:
    """Rolling window of past residuals (outcome - issued mean) for one target."""

    def __init__(self, method: str, window: int, min_n: int, default_sd: float, rng: tuple[int, int]):
        if method not in ("normal", "empirical"):
            raise ValueError("residual_method must be 'normal' or 'empirical'")
        self.method, self.min_n, self.default_sd, self.rng = method, min_n, default_sd, rng
        self.res: deque[float] = deque(maxlen=window)

    def add(self, r: float) -> None:
        self.res.append(r)

    def dist(self, mu: float) -> DiscreteDist:
        lo, hi = self.rng
        if len(self.res) < self.min_n:
            return from_normal(mu, self.default_sd, lo, hi)
        r = np.asarray(self.res)
        if self.method == "normal":
            return from_normal(mu, max(float(r.std()), 1.0), lo, hi)
        return from_residuals(mu, r, lo, hi)  # residuals kept as-is, so past bias carries over


class _SeasonCounter:
    def __init__(self) -> None:
        self._season_games: dict[str, int] = defaultdict(int)
        self.season: int | None = None

    def start_season(self, season: int) -> None:
        if self.season is not None and season != self.season:
            self._season_games.clear()
        self.season = season

    def _count(self, g: Game) -> None:
        self._season_games[g.home_key] += 1
        self._season_games[g.away_key] += 1

    def prior_games(self, team: str) -> int:
        return self._season_games[team]


class NaiveScores(_SeasonCounter):
    name = "scores_naive"
    feature_spec_version = "none"

    def __init__(self, margin_window: int = 2000, total_window: int = 512):
        super().__init__()
        self.margins: deque[int] = deque(maxlen=margin_window)
        self.totals: deque[int] = deque(maxlen=total_window)
        self.version = _version("0.1.0", {"mw": margin_window, "tw": total_window})

    def observe(self, g: Game) -> None:
        self._count(g)
        self.margins.append(g.home_score - g.away_score)
        self.totals.append(g.home_score + g.away_score)

    def predict_scores(self, g: Game) -> tuple[DiscreteDist, DiscreteDist]:
        m = from_residuals(0.0, np.asarray(self.margins), *MARGIN_RANGE) if self.margins else from_normal(2.0, 14.0, *MARGIN_RANGE)
        t = from_residuals(0.0, np.asarray(self.totals), *TOTAL_RANGE) if self.totals else from_normal(41.0, 14.0, *TOTAL_RANGE)
        return m, t

    def inputs_for(self, g: Game) -> dict:
        return {"n_margins": len(self.margins), "n_totals": len(self.totals)}


class EloScores:
    feature_spec_version = "elo_margin_v1"

    def __init__(self, elo_params: EloParams, *, residual_method: str = "normal", window: int = 2000,
                 min_samples: int = 200, total_window: int = 512, name: str | None = None):
        self.elo = EloModel(elo_params, name="elo_inner")
        self.name = name or f"scores_elo_{residual_method}"
        self.version = _version("0.1.0", {"elo": self.elo.version, "res": residual_method, "w": window,
                                          "min": min_samples, "tw": total_window})
        self.min_samples = min_samples
        self.margin_spread = _ResidualSpread(residual_method, window, min_samples, 13.5, MARGIN_RANGE)
        self.totals: deque[int] = deque(maxlen=total_window)
        self.total_spread = _ResidualSpread(residual_method, window, min_samples, 13.5, TOTAL_RANGE)
        self._pairs_x: list[float] = []
        self._pairs_y: list[float] = []
        self._coef = (0.0, 1.0 / 25.0)  # prior: 25 Elo points per point of margin, until fitted
        self.fit_season: int | None = None
        self._issued: dict[str, tuple[float, float, float]] = {}

    @property
    def season(self) -> int | None:
        return self.elo.season

    def start_season(self, season: int) -> None:
        self.elo.start_season(season)
        if len(self._pairs_x) >= self.min_samples and self.fit_season != season:
            x, y = np.asarray(self._pairs_x), np.asarray(self._pairs_y)
            b = float(np.cov(x, y, bias=True)[0, 1] / x.var()) if x.var() > 0 else 1 / 25
            self._coef = (float(y.mean() - b * x.mean()), b)
            self.fit_season = season

    def prior_games(self, team: str) -> int:
        return self.elo.prior_games(team)

    def _means(self, g: Game) -> tuple[float, float, float]:
        edge = self.elo._home_edge(g)  # Elo points, home-field advantage included (0 at neutral sites)
        mu_m = self._coef[0] + self._coef[1] * edge
        mu_t = float(np.mean(self.totals)) if self.totals else 41.0
        return edge, mu_m, mu_t

    def predict_scores(self, g: Game) -> tuple[DiscreteDist, DiscreteDist]:
        edge, mu_m, mu_t = self._means(g)
        self._issued[g.game_id] = (edge, mu_m, mu_t)
        return self.margin_spread.dist(mu_m), self.total_spread.dist(mu_t)

    def observe(self, g: Game) -> None:
        issued = self._issued.pop(g.game_id, None)
        margin, total = g.home_score - g.away_score, g.home_score + g.away_score
        if issued is not None:
            edge, mu_m, mu_t = issued
            self._pairs_x.append(edge)
            self._pairs_y.append(margin)
            self.margin_spread.add(margin - mu_m)
            self.total_spread.add(total - mu_t)
        self.totals.append(total)
        self.elo.observe(g)

    def inputs_for(self, g: Game) -> dict:
        edge, mu_m, mu_t = self._issued.get(g.game_id, (None, None, None))
        return {"elo_edge": edge, "margin_coef": [round(c, 5) for c in self._coef], "fit_season": self.fit_season}


@dataclass(frozen=True)
class PointsParams:
    k: float = 0.06               # rating step per point of residual
    k_league: float = 0.005       # league scoring-level step
    k_home: float = 0.003         # home-edge step
    season_regression: float = 0.4
    init_league: float = 20.5     # points per team per game
    init_home: float = 2.5        # points


class PointsModel(_SeasonCounter):
    """E[home pts] = L + off[h] - def[a] + H/2; E[away pts] = L + off[a] - def[h] - H/2 (H = 0 at neutral)."""

    feature_spec_version = "points_v1"

    def __init__(self, params: PointsParams | None = None, *, residual_method: str = "empirical",
                 window: int = 2000, min_samples: int = 200, name: str | None = None):
        super().__init__()
        self.p = params or PointsParams()
        self.name = name or f"scores_points_{residual_method}"
        self.version = _version("0.1.0", {"p": asdict(self.p), "res": residual_method, "w": window,
                                          "min": min_samples})
        self.off: dict[str, float] = defaultdict(float)
        self.deff: dict[str, float] = defaultdict(float)
        self.league = self.p.init_league
        self.home = self.p.init_home
        self.margin_spread = _ResidualSpread(residual_method, window, min_samples, 13.5, MARGIN_RANGE)
        self.total_spread = _ResidualSpread(residual_method, window, min_samples, 13.5, TOTAL_RANGE)
        self._issued: dict[str, tuple[float, float]] = {}

    def start_season(self, season: int) -> None:
        if self.season is not None and season > self.season:
            r = self.p.season_regression
            for t in list(self.off):
                self.off[t] *= 1 - r
                self.deff[t] *= 1 - r
        super().start_season(season)

    def expected_points(self, g: Game) -> tuple[float, float]:
        h = 0.0 if g.neutral_site else self.home
        eh = self.league + self.off[g.home_key] - self.deff[g.away_key] + h / 2
        ea = self.league + self.off[g.away_key] - self.deff[g.home_key] - h / 2
        return eh, ea

    def predict_scores(self, g: Game) -> tuple[DiscreteDist, DiscreteDist]:
        eh, ea = self.expected_points(g)
        mu_m, mu_t = eh - ea, eh + ea
        self._issued[g.game_id] = (mu_m, mu_t)
        return self.margin_spread.dist(mu_m), self.total_spread.dist(mu_t)

    def observe(self, g: Game) -> None:
        self._count(g)
        issued = self._issued.pop(g.game_id, None)
        if issued is not None:
            self.margin_spread.add((g.home_score - g.away_score) - issued[0])
            self.total_spread.add((g.home_score + g.away_score) - issued[1])
        eh, ea = self.expected_points(g)
        rh, ra = g.home_score - eh, g.away_score - ea
        k = self.p.k
        self.off[g.home_key] += k * rh
        self.deff[g.away_key] -= k * rh
        self.off[g.away_key] += k * ra
        self.deff[g.home_key] -= k * ra
        self.league += self.p.k_league * (rh + ra) / 2
        if not g.neutral_site:
            self.home += self.p.k_home * (rh - ra)

    def inputs_for(self, g: Game) -> dict:
        eh, ea = self.expected_points(g)
        return {"exp_home_pts": round(eh, 3), "exp_away_pts": round(ea, 3), "league": round(self.league, 3),
                "home_edge": round(self.home, 3)}
