"""`elo_qbadj`: Elo with a QB adjustment to each team's rating.

At forecast time, per team:
- expected starter: first QB in the depth-chart order (see `features/availability.py`) not listed Out or
  Doubtful on an injury report known by the replay clock; if no chart, the team's previous starter;
- QB rating: cumulative `qb_epa` per dropback from released games, shrunk toward `prior_mean` with
  `k_prior` dropbacks;
- baseline: dropback-weighted mean rating of the QBs who started for the team in released games this
  season (last season early on; the expected starter's own rating if neither exists);
- adjustment (Elo points) = `factor` x (expected starter rating - baseline). `factor` = 875 is fixed by
  reasoning: about 35 dropbacks per game x about 25 Elo points per point of margin.

The adjusted edge is used both for the forecast and as the expectation in the rating update (so a team
is not penalized for a loss its backup QB was expected to suffer). QB stats and starters are read only in
`observe()` (released games); injury rows only once known by the clock.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from typing import Any

from sfi.data.schema import Game
from sfi.features.availability import DepthCharts
from sfi.features.qb import InjuryRow
from sfi.models.elo import EloModel, EloParams


class EloQbAdj(EloModel):
    feature_spec_version = "elo_qbadj_v1"

    def __init__(self, params: EloParams, qb_games: dict[str, list[tuple[str, str, float, float]]],
                 starters: dict[tuple[str, str], str], injuries: dict[tuple[int, int, str], list[InjuryRow]],
                 depth: DepthCharts, *, factor: float = 875.0, k_prior: float = 300.0, prior_mean: float = 0.0,
                 name: str = "elo_qbadj", data_tag: str = ""):
        super().__init__(params, name=name)
        self._qb_games, self._starters, self._injuries, self.depth = qb_games, starters, injuries, depth
        self.factor, self.k_prior, self.prior_mean = float(factor), float(k_prior), float(prior_mean)
        spec = json.dumps({"elo": asdict_params(self.p), "factor": self.factor, "k": self.k_prior,
                           "m": self.prior_mean, "data": data_tag}, sort_keys=True).encode()
        self.version = f"0.1.0+{hashlib.sha256(spec).hexdigest()[:8]}"
        self.qb_epa: dict[str, float] = defaultdict(float)
        self.qb_db: dict[str, float] = defaultdict(float)
        self.last_starter: dict[str, str] = {}
        self.season_starts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))  # team -> qb -> dropbacks
        self.prev_season_starts: dict[str, dict[str, float]] = {}
        self.as_of: datetime | None = None
        self._adj: dict[str, tuple[float, float, dict]] = {}
        self._observing = False

    def set_as_of(self, as_of: datetime) -> None:
        self.as_of = as_of

    def start_season(self, season: int) -> None:
        if self.season is not None and season > self.season:
            self.prev_season_starts = {t: dict(d) for t, d in self.season_starts.items()}
            self.season_starts = defaultdict(lambda: defaultdict(float))
        super().start_season(season)

    def rating(self, qb: str | None) -> float:
        if qb is None:
            return self.prior_mean
        return (self.qb_epa[qb] + self.k_prior * self.prior_mean) / (self.qb_db[qb] + self.k_prior)

    def _out(self, g: Game, team: str) -> set[str]:
        if self.as_of is None:
            return set()
        return {r.player for r in self._injuries.get((g.season, g.week, team), ())
                if r.known_at <= self.as_of and r.status in ("Out", "Doubtful")}

    def team_side(self, g: Game, team: str) -> dict:
        order = self.depth.qb_order(g.season, g.week, team, self.as_of)
        out = self._out(g, team)
        prev = self.last_starter.get(team)
        if order:
            avail = [q for q in order if q not in out]
            expected = avail[0] if avail else None
            source = "depth_chart"
        else:
            expected = prev if prev not in out else None
            source = "previous_starter"
        base_pool = self.season_starts.get(team) or self.prev_season_starts.get(team) or {}
        if base_pool:
            tot = sum(base_pool.values())
            baseline = sum(self.rating(q) * n for q, n in base_pool.items()) / tot if tot > 0 else self.rating(expected)
        else:
            baseline = self.rating(expected)
        adj = self.factor * (self.rating(expected) - baseline)
        return {"expected": expected, "prev_starter": prev, "source": source, "listed_out": sorted(out & set(order or [prev])),
                "rating": round(self.rating(expected), 5), "baseline": round(baseline, 5), "adj_elo": round(adj, 2)}

    def _home_edge(self, game: Game) -> float:
        base = super()._home_edge(game)
        cached = self._adj.get(game.game_id)
        return base + (cached[0] - cached[1] if cached else 0.0)

    def predict_home_win(self, game: Game) -> float:
        if not self._observing:  # a forecast: compute the adjustment as of now and cache it
            h, a = self.team_side(game, game.home_key), self.team_side(game, game.away_key)
            self._adj[game.game_id] = (h["adj_elo"], a["adj_elo"], {"home": h, "away": a})
        return super().predict_home_win(game)

    def observe(self, game: Game) -> None:
        self._observing = True  # the rating update reuses the forecast-time adjustment (0 if never forecast)
        try:
            super().observe(game)
        finally:
            self._observing = False
        self._adj.pop(game.game_id, None)
        for qb, team, e, n in self._qb_games.get(game.game_id, ()):
            self.qb_epa[qb] += e
            self.qb_db[qb] += n
        for team in (game.home_key, game.away_key):
            s = self._starters.get((game.game_id, team))
            if s is not None:
                self.last_starter[team] = s
                drops = sum(n for q, t, _, n in self._qb_games.get(game.game_id, ()) if q == s and t == team)
                self.season_starts[team][s] += max(drops, 1.0)

    def inputs_for(self, game: Game) -> dict[str, Any]:
        cached = self._adj.get(game.game_id)
        return {**super().inputs_for(game), "factor": self.factor,
                **({"home_qb": cached[2]["home"], "away_qb": cached[2]["away"]} if cached else {})}


def asdict_params(p: EloParams) -> dict:
    from dataclasses import asdict
    return asdict(p)


class EloQbAdjAdaptiveHome(EloQbAdj):
    """`elo_qbadj` plus the learned home edge of `elo_hfa` (same rule: after each non-neutral game the edge
    moves by home_k x (outcome - expected), clipped to [0, 150], starting from the tuned 48)."""

    feature_spec_version = "elo_qbadj_hfa_v1"

    def __init__(self, params: EloParams, *args, home_k: float = 1.0, name: str = "elo_qbadj_hfa", **kw):
        super().__init__(params, *args, name=name, **kw)
        self.home_k = float(home_k)
        self.home_edge = self.p.home_field_elo
        self.version = self.version.replace("0.1.0+", "0.1.0-hfa" + f"{self.home_k:g}+")

    def _home_edge(self, game: Game) -> float:
        hfa = 0.0 if game.neutral_site else self.home_edge
        base = self.ratings[game.home_key] + hfa - self.ratings[game.away_key]
        cached = self._adj.get(game.game_id)
        return base + (cached[0] - cached[1] if cached else 0.0)

    def observe(self, game: Game) -> None:
        self._observing = True  # expectation with the forecast-time QB adjustment and current home edge
        try:
            expected = self.predict_home_win(game)
        finally:
            self._observing = False
        super().observe(game)
        if not game.neutral_site:
            self.home_edge = min(max(self.home_edge + self.home_k * (game.home_outcome - expected), 0.0), 150.0)

    def inputs_for(self, game: Game) -> dict[str, Any]:
        return {**super().inputs_for(game), "home_edge_elo": round(self.home_edge, 3), "home_k": self.home_k}
