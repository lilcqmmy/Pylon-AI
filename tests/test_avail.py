from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sfi.evaluation.walkforward import Replay
from sfi.features.availability import POS_TO_GROUP, WEIGHTS, DepthCharts, load_depth_charts
from sfi.features.qb import InjuryRow
from sfi.models.avail import EloAvail
from sfi.models.elo import EloParams

from .conftest import make_game

T0 = datetime(2015, 9, 13, 17, tzinfo=timezone.utc)


def test_position_groups_and_weight_order():
    assert POS_TO_GROUP["QB"] == "QB" and POS_TO_GROUP["LT"] == "OL" and POS_TO_GROUP["FS"] == "SECONDARY"
    assert POS_TO_GROUP["OLB"] == "PASS_RUSH" and POS_TO_GROUP["TE"] == "SKILL"
    assert "ILB" not in POS_TO_GROUP and "FB" not in POS_TO_GROUP and "K" not in POS_TO_GROUP
    w = WEIGHTS
    assert w["QB"] > w["OL"] == w["SECONDARY"] > w["PASS_RUSH"] > w["SKILL"]


def test_daily_depth_chart_uses_latest_snapshot_before_as_of():
    dc = DepthCharts()
    t1, t2 = datetime(2025, 10, 1, tzinfo=timezone.utc), datetime(2025, 10, 3, tzinfo=timezone.utc)
    dc.daily["KC"] = ([t1, t2], [{"q1": "QB"}, {"q2": "QB"}])
    assert dc.starters(2025, 5, "KC", datetime(2025, 10, 2, tzinfo=timezone.utc)) == {"q1": "QB"}
    assert dc.starters(2025, 5, "KC", datetime(2025, 10, 4, tzinfo=timezone.utc)) == {"q2": "QB"}
    assert dc.starters(2025, 5, "KC", datetime(2025, 9, 30, tzinfo=timezone.utc)) is None
    assert dc.starters(2025, 9, "KC", datetime(2025, 11, 30, tzinfo=timezone.utc)) is None  # stale (>14 days)


def test_weekly_chart_loader_dedupes_and_skips_special_teams(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    rows = {"season": [2015] * 5, "club_code": ["OAK"] * 5, "week": [3] * 5, "depth_team": ["1", "1", "1", "2", "1"],
            "formation": ["Offense", "Offense", "Defense", "Offense", "Special Teams"],
            "position": ["QB", "WR", "CB", "QB", "K"], "gsis_id": ["q", "w", "c", "q2", "k"]}
    path = tmp_path / "depth_charts_2015.parquet"
    pq.write_table(pa.table(rows), path)
    dc = load_depth_charts({2015: (path, {})})
    assert dc.weekly[(2015, 3, "LV")] == {"q": "QB", "w": "SKILL", "c": "SECONDARY"}  # franchise key, no K, no QB2


def _games(n=4):
    return [make_game(gid=f"2015_{w + 1:02d}_B_A", home="A", away="B", season=2015, week=w + 1,
                      start=T0 + timedelta(days=7 * w), hs=24, as_=17) for w in range(n)]


def test_injury_counts_only_starters_known_out_by_as_of():
    games = _games()
    target = games[2]
    ko = target.scheduled_start_utc
    dc = DepthCharts()
    dc.weekly[(2015, 3, "A")] = {"qa": "QB", "ta": "OL", "wa": "SKILL"}
    dc.weekly[(2015, 3, "B")] = {"qb": "QB"}
    inj = {(2015, 3, "A"): [InjuryRow("qa", "Out", ko - timedelta(hours=40), "date_modified"),
                            InjuryRow("ta", "Questionable", ko - timedelta(hours=40), "date_modified"),
                            InjuryRow("wa", "Doubtful", ko - timedelta(hours=10), "date_modified"),
                            InjuryRow("bench", "Out", ko - timedelta(hours=40), "date_modified")]}
    m = EloAvail(EloParams(), dc, inj)
    r = Replay(games, m)
    r.advance_to(ko - timedelta(hours=20))
    early = r.forecast(target).inputs["home"]
    assert early["out"]["QB"] == 1 and early["out"]["SKILL"] == 0 and early["n_out"] == 1  # WR report not yet known
    assert early["score"] == pytest.approx(WEIGHTS["QB"])
    r.advance_to(ko - timedelta(hours=1))
    late = r.forecast(target).inputs["home"]
    assert late["n_out"] == 2 and late["score"] == pytest.approx(WEIGHTS["QB"] + WEIGHTS["SKILL"])


def test_rare_availability_feature_is_not_standardized():
    m = EloAvail(EloParams(), DepthCharts(), {}, min_samples=10)
    rng = np.random.default_rng(0)
    for i in range(1000):
        m._X.append(np.array([rng.normal(), 3.0 if i == 5 else 0.0]))
        m._y.append(float(rng.uniform() < 0.5))
    m.elo.season = 2014
    m.start_season(2015)
    assert m._mean[1] == 0.0 and m._std[1] == 1.0 and abs(m._coef[2]) < 0.5
