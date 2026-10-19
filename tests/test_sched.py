import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from sfi.evaluation.walkforward import iter_walkforward
from sfi.features.schedule import feature_vector, miles, schedule_context
from sfi.models.elo import EloModel, EloParams
from sfi.models.sched import EloSched

from .conftest import make_game

T0 = datetime(2015, 9, 13, 17, tzinfo=timezone.utc)
STADIUMS = {"SEA00": {"lat": 47.5952, "lon": -122.3316}, "BOS00": {"lat": 42.0909, "lon": -71.2643},
            "LON00": {"lat": 51.5560, "lon": -0.2796}}


def _write(tmp_path, rows):
    path = tmp_path / "games.csv"
    cols = ["game_id", "stadium_id", "home_rest", "away_rest", "div_game"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def test_travel_rest_and_flags_from_schedule_only(tmp_path):
    g1 = make_game(gid="G1", home="SEA", away="NE", start=T0)                                  # Sunday 13:00 ET
    g2 = make_game(gid="G2", home="NE", away="SEA", start=T0 + timedelta(days=7, hours=7, minutes=20))  # SNF
    g3 = replace(make_game(gid="G3", home="SEA", away="NE", start=T0 + timedelta(days=14)), neutral_site=True)
    path = _write(tmp_path, [
        {"game_id": "G1", "stadium_id": "SEA00", "home_rest": "14", "away_rest": "4", "div_game": "0"},
        {"game_id": "G2", "stadium_id": "BOS00", "home_rest": "7", "away_rest": "7", "div_game": "1"},
        {"game_id": "G3", "stadium_id": "LON00", "home_rest": "NA", "away_rest": "7", "div_game": "0"},
    ])
    ctx = schedule_context(path, [g1, g2, g3], STADIUMS)
    c1 = ctx["G1"]
    assert c1["home_bye"] == 1.0 and c1["away_short"] == 1.0 and c1["rest_diff"] == 7  # 14-4 capped at 7
    assert c1["home_travel_mi"] == 0 and c1["away_travel_mi"] == pytest.approx(miles((42.0909, -71.2643), (47.5952, -122.3316)), rel=1e-3)
    assert 2400 < c1["away_travel_mi"] < 2600 and c1["away_tz"] == pytest.approx(51.07 / 15, abs=0.01)
    assert ctx["G2"]["primetime"] == 1.0 and ctx["G2"]["div"] == 1.0 and ctx["G1"]["primetime"] == 0.0
    c3 = ctx["G3"]  # neutral London: both teams travel; missing rest -> no rest difference
    assert c3["home_travel_mi"] > 4000 and c3["away_travel_mi"] > 3000 and c3["rest_diff"] == 0
    v = feature_vector(c1)
    assert v[0] == 7 and v[1] == 1.0 and v[2] == -1.0 and v[3] > 2.4  # home off bye, away short week, away travel


def test_context_ignores_scores(tmp_path):
    g = make_game(gid="G1", home="SEA", away="NE", start=T0, hs=30, as_=3)
    path = _write(tmp_path, [{"game_id": "G1", "stadium_id": "SEA00", "home_rest": "7", "away_rest": "7", "div_game": "0"}])
    a = schedule_context(path, [g], STADIUMS)
    b = schedule_context(path, [replace(g, home_score=0, away_score=50)], STADIUMS)
    assert a == b


def test_sched_model_is_elo_until_fitted_and_learns_only_from_past():
    games = [make_game(gid=f"G{i}", home="A", away="B", week=i + 1, start=T0 + timedelta(days=7 * i), hs=24, as_=10)
             for i in range(6)]
    ctx = {g.game_id: {"rest_diff": 3, "home_bye": 0.0, "away_bye": 0.0, "home_short": 0.0, "away_short": 0.0,
                       "home_travel_mi": 0.0, "away_travel_mi": 2000.0, "home_tz": 0.0, "away_tz": 2.0,
                       "primetime": 0.0, "div": 1.0} for g in games}
    m = EloSched(EloParams(), ctx, min_samples=10**6)
    a = [f.p_home_win for f in iter_walkforward(games, m, with_inputs=False)]
    b = [f.p_home_win for f in iter_walkforward(games, EloModel(EloParams()), with_inputs=False)]
    assert a == pytest.approx(b, abs=1e-12)
    m2 = EloSched(EloParams(), ctx, min_samples=1)
    list(iter_walkforward(games, m2))
    assert m2.fit_season is None  # one season only: no refit until a later season starts
