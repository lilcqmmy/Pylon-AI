"""Schedule-context features, computed from the schedule alone (published well before kickoff).

Per game: rest days (nflverse `home_rest`/`away_rest`), coming off a bye (rest >= 13), short week
(rest <= 4), rest difference, travel miles from each team's home stadium for that season to the venue,
approximate time zones crossed (|longitude difference| / 15), primetime (kickoff 19:00 ET or later) and
divisional game. No outcomes are used. Caveat: backtests use the final schedule; flexed kickoff times are
announced about 12 days ahead, before any forecast this project makes for those games.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from sfi.data.schema import Game
from sfi.data.sources.nflverse import EASTERN, NA


def miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


def _int(s: str) -> int | None:
    return None if s in NA else int(float(s))


def schedule_context(schedule_csv: Path, games: list[Game], stadiums: dict[str, dict]) -> dict[str, dict]:
    by_id = {g.game_id: g for g in games}
    rows = [r for r in csv.DictReader(open(schedule_csv, newline="", encoding="utf-8")) if r["game_id"] in by_id]
    # Each team's home venue per season: most frequent stadium among its non-neutral home games.
    home_counts: dict[tuple[int, str], Counter] = defaultdict(Counter)
    for r in rows:
        g = by_id[r["game_id"]]
        if not g.neutral_site and r.get("stadium_id"):
            home_counts[(g.season, g.home_key)][r["stadium_id"]] += 1
    home_venue = {k: c.most_common(1)[0][0] for k, c in home_counts.items()}

    def coords(sid):
        s = stadiums.get(sid or "")
        return None if s is None else (s["lat"], s["lon"])

    out = {}
    for r in rows:
        g = by_id[r["game_id"]]
        venue = coords(r.get("stadium_id"))
        ctx = {"venue_known": venue is not None}
        for side, key in (("home", g.home_key), ("away", g.away_key)):
            rest = _int(r.get(f"{side}_rest", "NA"))
            ctx[f"{side}_rest"] = rest
            ctx[f"{side}_bye"] = float(rest is not None and rest >= 13)
            ctx[f"{side}_short"] = float(rest is not None and rest <= 4)
            base = coords(home_venue.get((g.season, key)))
            if venue is None or base is None:
                ctx[f"{side}_travel_mi"], ctx[f"{side}_tz"] = 0.0, 0.0
            else:
                ctx[f"{side}_travel_mi"] = round(miles(base, venue), 1)
                ctx[f"{side}_tz"] = round(abs(base[1] - venue[1]) / 15.0, 2)
        hr, ar = ctx["home_rest"], ctx["away_rest"]
        ctx["rest_diff"] = 0 if hr is None or ar is None else max(min(hr - ar, 7), -7)
        ctx["primetime"] = float(g.scheduled_start_utc.astimezone(EASTERN).hour >= 19)
        ctx["div"] = float(r.get("div_game") == "1")
        out[g.game_id] = ctx
    return out


FEATURES = ["rest_diff", "bye_diff", "short_diff", "travel_diff_k", "tz_diff", "primetime", "div"]
CONTINUOUS = {"rest_diff", "travel_diff_k", "tz_diff"}


def feature_vector(ctx: dict) -> list[float]:
    """Home-minus-away differences (positive = favors the home side's situation) plus game-level flags."""
    return [
        float(ctx["rest_diff"]),
        ctx["home_bye"] - ctx["away_bye"],
        ctx["home_short"] - ctx["away_short"],
        (ctx["away_travel_mi"] - ctx["home_travel_mi"]) / 1000.0,
        ctx["away_tz"] - ctx["home_tz"],
        ctx["primetime"],
        ctx["div"],
    ]
