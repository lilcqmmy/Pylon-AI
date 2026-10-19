import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sfi.market import hypotheses as H
from sfi.market.dataset import build_rows, devig
from sfi.market.model import design, fit, walk_forward

from .conftest import make_game

T0 = datetime(2010, 9, 12, 17, tzinfo=timezone.utc)
COLS = ["game_id", "season", "game_type", "week", "spread_line", "total_line", "home_moneyline", "away_moneyline",
        "home_spread_odds", "away_spread_odds", "over_odds", "under_odds", "roof", "div_game", "home_rest", "away_rest"]


def _csv(tmp_path, rows):
    p = tmp_path / "games.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({**dict.fromkeys(COLS, "NA"), **r})
    return p


def _row(gid, season, week, spread, total=44.0, **kw):
    return {"game_id": gid, "season": str(season), "game_type": "REG", "week": str(week), "spread_line": str(spread),
            "total_line": str(total), "home_moneyline": "-150", "away_moneyline": "130", "home_spread_odds": "-110",
            "away_spread_odds": "-110", "over_odds": "-110", "under_odds": "-110", "roof": "outdoors",
            "div_game": "0", "home_rest": "7", "away_rest": "7", **kw}


def test_cover_push_over_and_devig(tmp_path):
    g = make_game(gid="G", home="A", away="B", season=2010, start=T0, hs=24, as_=21)  # margin 3
    p = _csv(tmp_path, [_row("G", 2010, 1, 3.0, total=45.0)])
    [x] = build_rows(p, [g], min_season=2006)
    assert x.home_cover() is None and x.over() is None  # margin 3 = spread 3 (push); total 45 = 45 (push)
    x2 = replace(x, spread=2.5, total=44.5)
    assert x2.home_cover() == 1.0 and x2.over() == 1.0
    assert devig(-110, -110) == pytest.approx(0.5) and devig(-150, 130) == pytest.approx(0.6 / (0.6 + 100 / 230))


def test_streak_uses_only_earlier_games(tmp_path):
    games, rows = [], []
    for w in range(5):  # A at home every week, always covers a 1-point spread by winning by 10
        g = make_game(gid=f"G{w}", home="A", away=f"X{w}", season=2010, week=w + 1, start=T0 + timedelta(days=7 * w),
                      hs=30, as_=20)
        games.append(g)
        rows.append(_row(f"G{w}", 2010, w + 1, 1.0))
    xs = build_rows(_csv(tmp_path, rows), games)
    assert [x.flags["home_streak3"] for x in xs] == [0.0, 0.0, 0.0, 1.0, 1.0]
    # Changing week-4's own result does not change its own streak flag
    games2 = [replace(g, home_score=0) if g.game_id == "G3" else g for g in games]
    xs2 = build_rows(_csv(tmp_path, rows), games2)
    assert xs2[3].flags["home_streak3"] == 1.0 and xs2[4].flags["home_streak3"] == 0.0


def test_high_total_threshold_uses_previous_seasons_only(tmp_path):
    rows, games = [], []
    i = 0
    for season, total in ((2007, 40.0), (2008, 40.0), (2009, 40.0)):
        for k in range(80):  # enough prior games (>= 200 over 3 seasons)
            rows.append(_row(f"P{i}", season, 1 + k % 17, -3.0, total=total + (k % 10)))
            i += 1
    g = make_game(gid="T", home="A", away="B", season=2010, start=T0, hs=20, as_=10)
    rows.append(_row("T", 2010, 1, -3.0, total=49.0))
    later = make_game(gid="L", home="C", away="D", season=2011, start=T0 + timedelta(days=365), hs=20, as_=10)
    rows.append(_row("L", 2011, 1, -3.0, total=60.0))
    xs = {x.game.game_id: x for x in build_rows(_csv(tmp_path, rows), [g, later])}
    assert xs["T"].flags["high_total"] == 1.0  # 49 >= 90th pct of 40..49 from 2007-2009 (= 49)
    rows[-2] = _row("T", 2010, 1, -3.0, total=48.0)
    xs = {x.game.game_id: x for x in build_rows(_csv(tmp_path, rows), [g, later])}
    assert xs["T"].flags["high_total"] == 0.0


def test_home_dog_flags_exclude_neutral_sites(tmp_path):
    g = replace(make_game(gid="N", home="A", away="B", season=2010, start=T0, hs=20, as_=10), neutral_site=True)
    [x] = build_rows(_csv(tmp_path, [_row("N", 2010, 1, -4.0)]), [g])
    assert x.flags["home_dog"] == 0.0 and x.flags["road_fav7"] == 0.0


def test_market_slope_is_not_shrunk_by_the_ridge():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 6000)
    y = (rng.uniform(size=z.size) < 1 / (1 + np.exp(-z))).astype(float)
    X = np.column_stack([np.ones_like(z), z, rng.integers(0, 2, z.size)])
    beta = fit(X, y, l2=1e6)  # huge penalty on the indicator only
    assert beta[1] == pytest.approx(1.0, abs=0.08) and abs(beta[2]) < 1e-3


def test_walk_forward_predictions_ignore_later_seasons(tmp_path):
    rows, games = [], []
    rng = np.random.default_rng(1)
    for s in (2006, 2007, 2008, 2009):
        for k in range(60):
            gid = f"{s}_{k}"
            hs, as_ = int(rng.integers(10, 35)), int(rng.integers(10, 35))
            games.append(make_game(gid=gid, home=f"H{k % 8}", away=f"A{k % 8}", season=s, week=1 + k // 4,
                                   start=datetime(s, 9, 10, tzinfo=timezone.utc) + timedelta(days=k), hs=hs, as_=as_))
            rows.append(_row(gid, s, 1 + k // 4, float(rng.choice([-7, -3, 1, 3, 6.5]))))
    xs = build_rows(_csv(tmp_path, rows), games)
    a = walk_forward(xs, "cover", [2006, 2007, 2008, 2009], l2=10.0)
    flipped = [replace(x, game=replace(x.game, home_score=x.game.away_score, away_score=x.game.home_score))
               if x.game.season == 2009 else x for x in xs]
    b = walk_forward(flipped, "cover", [2006, 2007, 2008, 2009], l2=10.0)
    n08 = sum(1 for x in a.rows if x.game.season == 2008)
    assert n08 > 0 and np.allclose(a.p_model[:n08], b.p_model[:n08])


def test_hypothesis_selection_sides():
    from sfi.market.dataset import MarketRow
    g = make_game(gid="G", home="A", away="B", season=2010, start=T0, hs=17, as_=20)  # home loses by 3
    base = dict(game=g, spread=-7.0, total=40.0, home_ml=200, away_ml=-240, home_spread_odds=-110,
                away_spread_odds=-110, over_odds=-110, under_odds=-110, p_home_ml=0.3, p_home_cover=0.5,
                p_over=0.5, roof="outdoors")
    flags = {k: 0.0 for k in ("home_dog", "home_dog_prime", "late_home_dog", "road_fav7", "high_total", "div",
                              "home_bye_only", "away_bye_only", "home_short_only", "away_short_only", "key",
                              "primetime", "home_streak3", "away_streak3")}
    flags.update(home_dog=1.0, road_fav7=1.0, key=1.0, dome=0.0)
    x = MarketRow(**base, flags=flags)
    b = H.select([x], "b")["main"]
    assert b.hits == [1.0] and b.returns[0] == pytest.approx(100 / 110)  # +7 home dog lost by 3: covers
    f = H.select([x], "f")["main"]
    assert f.hits == [1.0]  # underdog (home) at a key number covers
