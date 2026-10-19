"""Per-game market rows: lines (treated as the close), prices, outcomes, and the pre-registered hypothesis
indicators (docs/market_biases.md). All indicators are knowable before kickoff: schedule fields, the line itself,
totals from earlier seasons, and ATS results of earlier games."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sfi.data.schema import Game
from sfi.data.sources.nflverse import EASTERN, NA


def implied(odds: float) -> float:
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def devig(a: float | None, b: float | None) -> float | None:
    """Probability of side a from two American prices with the margin removed proportionally."""
    if a is None or b is None or a == 0 or b == 0:
        return None
    pa, pb = implied(a), implied(b)
    return pa / (pa + pb)


def _num(r: dict, k: str) -> float | None:
    v = r.get(k, "NA")
    if v in NA:
        return None
    try:
        return float(v)
    except ValueError:
        return None


@dataclass
class MarketRow:
    game: Game
    spread: float                 # home favored by this many points
    total: float | None
    home_ml: float | None
    away_ml: float | None
    home_spread_odds: float
    away_spread_odds: float
    over_odds: float
    under_odds: float
    p_home_ml: float | None       # de-vigged moneyline P(home win)
    p_home_cover: float           # de-vigged spread prices (0.5 if missing)
    p_over: float                 # de-vigged total prices (0.5 if missing)
    roof: str
    flags: dict = field(default_factory=dict)

    @property
    def margin(self) -> int | None:
        g = self.game
        return None if not g.is_final else g.home_score - g.away_score

    def home_cover(self) -> float | None:
        """1 home covers, 0 away covers, None push or not final."""
        m = self.margin
        if m is None or m == self.spread:
            return None
        return 1.0 if m > self.spread else 0.0

    def over(self) -> float | None:
        g = self.game
        if not g.is_final or self.total is None:
            return None
        t = g.home_score + g.away_score
        return None if t == self.total else (1.0 if t > self.total else 0.0)


def build_rows(schedule_csv: Path, games: list[Game], *, min_season: int = 2006) -> list[MarketRow]:
    """Rows for final or upcoming games from `min_season` on (only those in `games`), with indicators."""
    by_id = {g.game_id: g for g in games}
    raw = {r["game_id"]: r for r in csv.DictReader(open(schedule_csv, newline="", encoding="utf-8"))}
    # point-in-time inputs that use other games
    totals_by_season: dict[int, list[float]] = defaultdict(list)
    last_reg_week: dict[int, int] = {}
    for r in raw.values():
        s = int(r["season"])
        t = _num(r, "total_line")
        if t is not None:
            totals_by_season[s].append(t)
        if r["game_type"] == "REG":
            last_reg_week[s] = max(last_reg_week.get(s, 0), int(r["week"]))

    def high_total_threshold(season: int) -> float | None:
        prev = [t for s in (season - 3, season - 2, season - 1) for t in totals_by_season.get(s, [])]
        return float(np.percentile(prev, 90)) if len(prev) >= 200 else None

    rows: list[MarketRow] = []
    for gid, g in sorted(by_id.items(), key=lambda kv: (kv[1].scheduled_start_utc, kv[0])):
        r = raw.get(gid)
        if r is None or g.season < min_season:
            continue
        s = _num(r, "spread_line")
        if s is None:
            continue
        hso, aso = _num(r, "home_spread_odds") or -110.0, _num(r, "away_spread_odds") or -110.0
        oo, uo = _num(r, "over_odds") or -110.0, _num(r, "under_odds") or -110.0
        rows.append(MarketRow(
            game=g, spread=s, total=_num(r, "total_line"), home_ml=_num(r, "home_moneyline"),
            away_ml=_num(r, "away_moneyline"), home_spread_odds=hso, away_spread_odds=aso, over_odds=oo,
            under_odds=uo, p_home_ml=devig(_num(r, "home_moneyline"), _num(r, "away_moneyline")),
            p_home_cover=devig(hso, aso) or 0.5, p_over=devig(oo, uo) or 0.5,
            roof="" if r.get("roof") in NA else r["roof"],
        ))
        x = rows[-1]
        f = x.flags
        neutral = g.neutral_site
        prime = g.scheduled_start_utc.astimezone(EASTERN).hour >= 19
        f["home_dog"] = float(not neutral and s < 0)
        f["home_dog_prime"] = float(f["home_dog"] and prime)
        f["late_home_dog"] = float(f["home_dog"] and r["game_type"] == "REG"
                                   and int(r["week"]) >= last_reg_week.get(g.season, 99) - 3)
        f["road_fav7"] = float(not neutral and s <= -7)
        f["dome"] = 1.0 if x.roof == "dome" else (0.0 if x.roof == "outdoors" else np.nan)
        thr = high_total_threshold(g.season)
        f["high_total"] = float(thr is not None and x.total is not None and x.total >= thr)
        f["div"] = float(r.get("div_game") == "1")
        hr, ar = _num(r, "home_rest"), _num(r, "away_rest")
        hb, ab = (hr or 0) >= 13, (ar or 0) >= 13
        hs, as_ = hr is not None and hr <= 4, ar is not None and ar <= 4
        f["home_bye_only"], f["away_bye_only"] = float(hb and not ab), float(ab and not hb)
        f["home_short_only"], f["away_short_only"] = float(hs and not as_), float(as_ and not hs)
        f["key"] = float(abs(s) in (3.0, 7.0))
        f["primetime"] = float(prime)
    _add_streaks(rows)
    return rows


def _add_streaks(rows: list[MarketRow]) -> None:
    """flags home_streak3 / away_streak3: the team covered each of its previous 3 games (same season).
    Uses only games that kicked off earlier (results known long before the next game)."""
    history: dict[tuple[int, str], list[float]] = defaultdict(list)  # (season, team) -> cover sequence
    for x in rows:  # rows are in kickoff order
        g = x.game
        for side, team in (("home", g.home_key), ("away", g.away_key)):
            seq = history[(g.season, team)]
            x.flags[f"{side}_streak3"] = float(len(seq) >= 3 and all(c == 1.0 for c in seq[-3:]))
        hc = x.home_cover()
        m = x.margin
        if m is not None:
            history[(g.season, g.home_key)].append(0.5 if hc is None else hc)
            history[(g.season, g.away_key)].append(0.5 if hc is None else 1.0 - hc)
