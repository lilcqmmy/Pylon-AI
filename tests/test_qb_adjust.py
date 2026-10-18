from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sfi.evaluation.walkforward import Replay, iter_walkforward
from sfi.features.qb import InjuryRow, load_injuries
from sfi.models.elo import EloParams
from sfi.models.qb_adjust import QbAdjusted

from .conftest import make_game

T0 = datetime(2015, 9, 13, 17, tzinfo=timezone.utc)


def _season(n_weeks=6):
    """A and B play each week; A's starter is 'qbA', backup 'qbA2'; B always 'qbB'."""
    games, qb_games, starters = [], {}, {}
    for w in range(n_weeks):
        g = make_game(gid=f"2015_{w + 1:02d}_B_A", home="A", away="B", season=2015, week=w + 1,
                      start=T0 + timedelta(days=7 * w), hs=24, as_=17)
        games.append(g)
        qb_games[g.game_id] = [("qbA", "A", 6.0, 30.0), ("qbA2", "A", -1.0, 4.0), ("qbB", "B", 0.0, 30.0)]
        starters[(g.game_id, "A")] = "qbA"
        starters[(g.game_id, "B")] = "qbB"
    return games, qb_games, starters


def test_injury_row_used_only_once_known_at_as_of():
    games, qb_games, starters = _season()
    target = games[3]
    ko = target.scheduled_start_utc
    inj = {(2015, target.week, "A"): [InjuryRow("qbA", "Out", ko - timedelta(hours=10), "date_modified")]}
    m = QbAdjusted(EloParams(), qb_games, starters, inj)
    r = Replay(games, m)
    r.advance_to(ko - timedelta(hours=12))  # report not yet known
    early = r.forecast(target).inputs["home"]
    assert early["status"] is None and early["expected_starter"] == "qbA"
    r.advance_to(ko - timedelta(hours=1))   # report known
    late = r.forecast(target).inputs["home"]
    assert late["status"] == "Out" and late["expected_starter"] == "qbA2" and late["rating_delta"] < 0


def test_target_game_starter_and_qb_stats_are_never_read_before_release():
    games, qb_games, starters = _season()
    target = games[3]

    def run(st, qg):
        m = QbAdjusted(EloParams(), qg, st, {}, min_samples=1)
        return {f.game.game_id: (f.p_home_win, f.inputs) for f in iter_walkforward(games, m)}

    base = run(starters, qb_games)
    st2 = dict(starters)
    st2[(target.game_id, "A")] = "someone_else"
    qg2 = dict(qb_games)
    qg2[target.game_id] = [("qbA", "A", -500.0, 30.0)]
    after = run(st2, qg2)
    for g in games[:4]:  # up to and including the target, nothing changes
        assert after[g.game_id] == base[g.game_id], g.game_id
    assert after[games[4].game_id][1]["home"]["prev_starter"] == "someone_else"


def test_rare_flags_are_not_standardized_into_huge_adjustments():
    rng = np.random.default_rng(0)
    m = QbAdjusted(EloParams(), {}, {}, {}, min_samples=10)
    for i in range(2000):
        x = np.array([rng.normal(), 0.0, 1.0 if i == 7 else 0.0, 0.0, 0.0, 0.0])
        m._X.append(x)
        m._y.append(float(rng.uniform() < 1 / (1 + np.exp(-x[0]))))
    m._refit()
    assert m._mean[2] == 0.0 and m._std[2] == 1.0  # flag column left raw
    # One flagged training game cannot create a large effect under the ridge penalty.
    assert abs(m._coef[3]) < 0.5


def test_replay_passes_clock_to_models():
    games, qb_games, starters = _season(2)
    m = QbAdjusted(EloParams(), qb_games, starters, {})
    r = Replay(games, m)
    t = games[1].scheduled_start_utc - timedelta(hours=2)
    r.advance_to(t)
    assert m.as_of == t


def test_load_injuries_timestamps_policy_and_drops(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    g = make_game(gid="2015_03_B_A", home="A", away="B", season=2015, week=3, start=T0)
    rows = {
        "season": [2015, 2015, 2015, 2015, 2015], "week": [3, 3, 3, 3, 9],
        "team": ["A", "A", "A", "B", "A"], "gsis_id": ["q1", "q2", "q3", "w1", "q1"],
        "position": ["QB", "QB", "QB", "WR", "QB"], "report_status": ["Out", "Questionable", "Out", "Out", "Out"],
        "date_modified": [T0 - timedelta(hours=50), None, T0 + timedelta(minutes=5), T0 - timedelta(hours=50),
                          T0 - timedelta(hours=50)],
    }
    path = tmp_path / "injuries_2015.parquet"
    pq.write_table(pa.table(rows), path)
    res = load_injuries({2015: (path, {})}, [g], missing_hours=24)
    used = res.rows[(2015, 3, "A")]
    assert [(r.player, r.timestamp_source) for r in used] == [("q1", "date_modified"), ("q2", "policy")]
    assert used[1].known_at == T0 - timedelta(hours=24)
    assert res.dropped_after_kickoff == 1 and res.policy_rows == 1 and res.unmatched == 1  # week 9 has no game
