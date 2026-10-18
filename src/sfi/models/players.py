"""Player yardage distributions (passing, rushing, receiving), learned walk-forward.

Pregame player pool for a team and stat: players with at least `min_opp` opportunities for that team
in its last `pool_games` released games, still on that team (their latest released appearance was for
it), and not listed Out/Doubtful on an injury report known by the replay clock. For passing, only the
expected starter (most attempts in the team's latest game among eligible QBs) is forecast. A pool player who then
records no opportunity is scored as 0 yards: the evaluation never conditions on who actually played.

Models (same pool, same spread method; differ only in the mean):
- `PlayerRecentAvg`: mean yards over the player's last 4 appearances in that stat.
- `PlayerOppEff`: recent volume (weighted mean of opportunities) x player yards per opportunity (career,
  shrunk to the league rate) x opponent adjustment (yards per opportunity the opponent allowed this
  season, shrunk to league; last season early on).

Spread: residuals (outcome - issued mean) of the model's own past forecasts for the same stat, taken from
forecasts with a similar mean (bins expanding outward until `min_k` residuals), so skew and the mass at
0 yards for pool players who don't play are represented. Point-in-time: player stats are read only in
`observe()` (released games); injury rows only once known by the clock.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

from sfi.data.schema import Game
from sfi.distributions import DiscreteDist, from_normal, from_residuals
from sfi.features.players import STATS, PlayerGames
from sfi.features.qb import InjuryRow

SUPPORT = {"pass": (-20, 600), "rush": (-30, 300), "rec": (-10, 300)}
BIN_WIDTH = {"pass": 25.0, "rush": 10.0, "rec": 10.0}


@dataclass(frozen=True)
class PlayerParams:
    pool_games: int = 2
    min_opp: dict = field(default_factory=lambda: {"pass": 10, "rush": 3, "rec": 2})
    k_player: dict = field(default_factory=lambda: {"pass": 150.0, "rush": 80.0, "rec": 40.0})
    k_def: dict = field(default_factory=lambda: {"pass": 300.0, "rush": 150.0, "rec": 300.0})
    volume_decay: float = 0.35
    residual_window: int = 4000   # per bin
    min_k: int = 300


@dataclass
class PlayerForecast:
    game: Game
    player: str
    team: str
    stat: str
    mu: float
    dist: DiscreteDist
    inputs: dict


class _PlayerBase:
    kind = "base"

    def __init__(self, stats: PlayerGames, injuries: dict[tuple[int, int, str], list[InjuryRow]] | None = None,
                 params: PlayerParams | None = None, name: str | None = None, data_tag: str = ""):
        self.p = params or PlayerParams()
        self._stats, self._injuries = stats, injuries or {}
        self.name = name or f"players_{self.kind}"
        self.version = "0.2.0+" + hashlib.sha256(json.dumps(
            {"kind": self.kind, "p": asdict(self.p), "data": data_tag}, sort_keys=True).encode()).hexdigest()[:8]
        self.feature_spec_version = f"players_{self.kind}_v1"
        self.season: int | None = None
        self.as_of: datetime | None = None
        self.recent: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=8))   # (player, stat) -> (opp, yds)
        self.career: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
        self.last_team: dict[str, str] = {}
        self.team_games: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.p.pool_games))  # team -> [{(p, stat): opp}]
        self.league = {s: [0.0, 0.0] for s in STATS}
        self.def_season: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
        self.def_prior: dict[tuple[str, str], list[float]] = {}
        self.res_bins: dict[tuple[str, int], deque] = defaultdict(lambda: deque(maxlen=self.p.residual_window))
        self._issued: dict[str, list[tuple[str, str, float]]] = {}

    # ---- replay interface
    def set_as_of(self, as_of: datetime) -> None:
        self.as_of = as_of

    def start_season(self, season: int) -> None:
        if self.season is not None and season > self.season:
            self.def_prior = {k: list(v) for k, v in self.def_season.items()}
            self.def_season.clear()
        self.season = season

    def prior_games(self, team: str) -> int:
        return 0

    def observe(self, g: Game) -> None:
        rows = self._stats.get(g.game_id, [])
        got = {(p, s): y for p, _, s, _, y in rows}
        for player, stat, mu in self._issued.pop(g.game_id, []):
            y = got.get((player, stat), 0.0)  # in the pool but no opportunity: 0 yards
            self.res_bins[(stat, self._bin(stat, mu))].append(y - mu)
        per_team: dict[str, dict] = {g.home_key: {}, g.away_key: {}}
        for player, team, stat, n, y in rows:
            self.recent[(player, stat)].append((n, y))
            c = self.career[(player, stat)]
            c[0] += n
            c[1] += y
            self.last_team[player] = team
            self.league[stat][0] += n
            self.league[stat][1] += y
            if team in per_team:
                per_team[team][(player, stat)] = n
                opp = g.away_key if team == g.home_key else g.home_key
                d = self.def_season[(opp, stat)]
                d[0] += n
                d[1] += y
        for team, usage in per_team.items():
            self.team_games[team].append(usage)

    # ---- pool and spread
    def _bin(self, stat: str, mu: float) -> int:
        return int(max(mu, 0.0) // BIN_WIDTH[stat])

    def _out(self, g: Game, team: str) -> set[str]:
        if self.as_of is None:
            return set()
        return {r.player for r in self._injuries.get((g.season, g.week, team), ())
                if r.known_at <= self.as_of and r.status in ("Out", "Doubtful")}

    def pool(self, g: Game, team: str) -> list[tuple[str, str]]:
        usage: dict[tuple[str, str], float] = defaultdict(float)
        for game_usage in self.team_games.get(team, ()):
            for key, n in game_usage.items():
                usage[key] += n
        out = self._out(g, team)
        pool = [k for k, n in usage.items()
                if n >= self.p.min_opp[k[1]] and self.last_team.get(k[0]) == team and k[0] not in out]
        # Passing: one expected starter, not every QB who threw recently (backups who mopped up would
        # otherwise make the forecasts bimodal). Starter = most attempts in the team's latest game among
        # eligible QBs; if he is out, the next eligible QB.
        games = list(self.team_games.get(team, ()))
        qbs = [k for k in pool if k[1] == "pass"]
        if len(qbs) > 1 and games:
            def recency_key(k):
                return tuple(-gu.get(k, 0.0) for gu in reversed(games))
            starter = min(qbs, key=recency_key)
            pool = [k for k in pool if k[1] != "pass"] + [starter]
        return sorted(pool)

    def spread(self, stat: str, mu: float) -> DiscreteDist:
        lo, hi = SUPPORT[stat]
        b = self._bin(stat, mu)
        pool: list[float] = list(self.res_bins.get((stat, b), ()))
        r = 1
        max_bin = int(hi // BIN_WIDTH[stat]) + 1
        while len(pool) < self.p.min_k and r <= max_bin:
            for nb in (b - r, b + r):
                if nb >= 0:
                    pool.extend(self.res_bins.get((stat, nb), ()))
            r += 1
        if len(pool) < self.p.min_k:
            return from_normal(mu, max(8.0, 0.6 * abs(mu)), lo, hi)
        return from_residuals(mu, np.asarray(pool), lo, hi)

    # ---- forecasting
    def mean(self, g: Game, player: str, team: str, stat: str) -> tuple[float, dict]:
        raise NotImplementedError

    def predict_players(self, g: Game) -> list[PlayerForecast]:
        out, issued = [], []
        for team in (g.home_key, g.away_key):
            for player, stat in self.pool(g, team):
                mu, info = self.mean(g, player, team, stat)
                issued.append((player, stat, mu))
                out.append(PlayerForecast(g, player, team, stat, mu, self.spread(stat, mu), info))
        self._issued[g.game_id] = issued
        return out

    def league_rate(self, stat: str) -> float:
        n, y = self.league[stat]
        return y / n if n > 0 else {"pass": 6.5, "rush": 4.2, "rec": 7.5}[stat]


class PlayerRecentAvg(_PlayerBase):
    kind = "recent_avg"

    def mean(self, g, player, team, stat):
        hist = list(self.recent[(player, stat)])[-4:]
        mu = float(np.mean([y for _, y in hist])) if hist else 0.0
        return mu, {"n_recent": len(hist)}


class PlayerOppEff(_PlayerBase):
    kind = "opp_eff"

    def mean(self, g, player, team, stat):
        hist = list(self.recent[(player, stat)])
        w = np.array([(1 - self.p.volume_decay) ** i for i in range(len(hist))][::-1])
        vol = float(np.dot(w, [n for n, _ in hist]) / w.sum()) if hist else 0.0
        lg = self.league_rate(stat)
        cn, cy = self.career[(player, stat)]
        kp = self.p.k_player[stat]
        ypo = (cy + kp * lg) / (cn + kp)
        opp = g.away_key if team == g.home_key else g.home_key
        d = self.def_season.get((opp, stat))
        if not d or d[0] == 0:
            d = self.def_prior.get((opp, stat), [0.0, 0.0])
        kd = self.p.k_def[stat]
        adj = ((d[1] + kd * lg) / (d[0] + kd)) / lg if lg > 0 else 1.0
        mu = vol * ypo * adj
        return mu, {"volume": round(vol, 3), "ypo": round(ypo, 3), "def_adj": round(adj, 4)}
