from datetime import datetime, timedelta, timezone

import pytest

from sfi.evaluation.walkforward import Replay, iter_walkforward
from sfi.features.availability import DepthCharts
from sfi.features.qb import InjuryRow
from sfi.models.elo import EloModel, EloParams
from sfi.models.qbadj import EloQbAdj

from .conftest import make_game

T0 = datetime(2015, 9, 13, 17, tzinfo=timezone.utc)


def _games(n=4):
    return [make_game(gid=f"2015_{w + 1:02d}_B_A", home="A", away="B", season=2015, week=w + 1,
                      start=T0 + timedelta(days=7 * w), hs=24, as_=17) for w in range(n)]


def _data(games):
    qb_games = {g.game_id: [("qa", "A", 9.0, 30.0), ("qb", "B", 0.0, 30.0)] for g in games}  # qa strong
    starters = {}
    for g in games:
        starters[(g.game_id, "A")] = "qa"
        starters[(g.game_id, "B")] = "qb"
    depth = DepthCharts()
    for g in games:
        depth.qb_weekly[(2015, g.week, "A")] = ["qa", "qa2"]
        depth.qb_weekly[(2015, g.week, "B")] = ["qb"]
    return qb_games, starters, depth


def test_expected_starter_switches_only_when_out_is_known_and_lowers_rating():
    games = _games()
    qb_games, starters, depth = _data(games)
    target = games[3]
    ko = target.scheduled_start_utc
    inj = {(2015, 4, "A"): [InjuryRow("qa", "Out", ko - timedelta(hours=30), "date_modified")]}
    m = EloQbAdj(EloParams(), qb_games, starters, inj, depth)
    r = Replay(games, m)
    r.advance_to(ko - timedelta(hours=40))
    early = r.forecast(target)
    assert early.inputs["home_qb"]["expected"] == "qa" and early.inputs["home_qb"]["adj_elo"] == pytest.approx(0.0)
    r.advance_to(ko - timedelta(hours=1))
    late = r.forecast(target)
    h = late.inputs["home_qb"]
    assert h["expected"] == "qa2" and h["adj_elo"] < -50  # backup (no history) rated at the prior, below qa
    assert late.p_home_win < early.p_home_win


def test_falls_back_to_previous_starter_without_chart():
    games = _games()
    qb_games, starters, _ = _data(games)
    m = EloQbAdj(EloParams(), qb_games, starters, {}, DepthCharts())
    fc = {f.game.game_id: f for f in iter_walkforward(games, m)}
    assert fc[games[2].game_id].inputs["home_qb"]["source"] == "previous_starter"
    assert fc[games[2].game_id].inputs["home_qb"]["expected"] == "qa"


def test_no_adjustment_equals_plain_elo():
    games = _games()
    m = EloQbAdj(EloParams(), {}, {}, {}, DepthCharts())
    a = [f.p_home_win for f in iter_walkforward(games, m, with_inputs=False)]
    b = [f.p_home_win for f in iter_walkforward(games, EloModel(EloParams()), with_inputs=False)]
    assert a == pytest.approx(b, abs=1e-12)


def test_unreleased_qb_stats_and_starters_are_not_read():
    games = _games(5)
    qb_games, starters, depth = _data(games)
    target = games[3]

    def run(qg, st):
        m = EloQbAdj(EloParams(), qg, st, {}, depth)
        return {f.game.game_id: (f.p_home_win, f.inputs.get("home_qb")) for f in iter_walkforward(games, m)}

    base = run(qb_games, starters)
    qg2 = {**qb_games, target.game_id: [("qa", "A", -900.0, 30.0)]}
    st2 = {**starters, (target.game_id, "A"): "zz"}
    after = run(qg2, st2)
    for g in games[:4]:
        assert after[g.game_id] == base[g.game_id], g.game_id
    assert after[games[4].game_id] != base[games[4].game_id]


def test_version_distinct_from_frozen_elo():
    p = EloParams(k_factor=20, home_field_elo=48, season_regression=0.5)
    assert EloQbAdj(p, {}, {}, {}, DepthCharts()).version != EloModel(p).version == "0.3.0+851aa1d0"


def test_combo_without_signals_equals_adaptive_home_elo():
    from sfi.models.avail import EloAvail
    from sfi.models.elo import EloAdaptiveHome
    from sfi.models.qbadj import EloQbAdjAdaptiveHome
    from sfi.features.availability import WEIGHTS
    games = _games(6)
    p = EloParams(k_factor=20, home_field_elo=48, season_regression=0.5)
    base = EloQbAdjAdaptiveHome(p, {}, {}, {}, DepthCharts(), home_k=1.0)
    combo = EloAvail(p, DepthCharts(), {}, weights={**WEIGHTS, "QB": 0.0}, min_samples=10**6, name="elo_combo", base=base)
    ah = EloAdaptiveHome(p, home_k=1.0)
    a = [f.p_home_win for f in iter_walkforward(games, combo, with_inputs=False)]
    b = [f.p_home_win for f in iter_walkforward(games, ah, with_inputs=False)]
    assert a == pytest.approx(b, abs=1e-12)
    assert combo.elo.home_edge == pytest.approx(ah.home_edge) and ah.home_edge != 48  # learned identically


def test_combo_ignores_qb_in_availability_but_counts_other_groups():
    from sfi.models.avail import EloAvail
    from sfi.models.qbadj import EloQbAdjAdaptiveHome
    from sfi.features.availability import WEIGHTS
    games = _games(4)
    target = games[2]
    ko = target.scheduled_start_utc
    dc = DepthCharts()
    dc.weekly[(2015, 3, "A")] = {"qa": "QB", "ta": "OL"}
    dc.qb_weekly[(2015, 3, "A")] = ["qa", "qa2"]
    inj = {(2015, 3, "A"): [InjuryRow("qa", "Out", ko - timedelta(hours=30), "date_modified"),
                            InjuryRow("ta", "Out", ko - timedelta(hours=30), "date_modified")]}
    qb_games = {g.game_id: [("qa", "A", 9.0, 30.0)] for g in games}
    starters = {(g.game_id, "A"): "qa" for g in games}
    p = EloParams()
    base = EloQbAdjAdaptiveHome(p, qb_games, starters, inj, dc)
    combo = EloAvail(p, dc, inj, weights={**WEIGHTS, "QB": 0.0}, name="elo_combo", base=base)
    r = Replay(games, combo)
    r.advance_to(ko - timedelta(hours=1))
    f = r.forecast(target)
    assert f.inputs["home"]["out"]["QB"] == 1 and f.inputs["home"]["score"] == pytest.approx(WEIGHTS["OL"])
    assert f.inputs["base"]["home_qb"]["expected"] == "qa2" and f.inputs["base"]["home_qb"]["adj_elo"] < 0
