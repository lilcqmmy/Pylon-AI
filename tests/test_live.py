from dataclasses import replace
from datetime import timedelta

import pytest

from sfi import live
from sfi.evaluation.walkforward import iter_walkforward
from sfi.models.baselines import HomeRate
from sfi.models.elo import EloModel

from .conftest import T0, make_game

CFG = live.LiveConfig(horizon_days=8, min_lead_minutes=10, max_snapshot_age_hours=24, known_at_hours=4)
MANIFEST = {"retrieved_at_utc": live._iso(T0 + timedelta(days=6)), "upstream_last_updated": "x"}


def _season():
    """Two weeks of results, then a third week unplayed."""
    gs = []
    for w in range(3):
        start = T0 + timedelta(days=7 * w)
        played = w < 2
        gs.append(make_game(gid=f"W{w}A", home="A", away="B", week=w + 1, start=start,
                            hs=24 if played else None, as_=10 if played else None))
        gs.append(make_game(gid=f"W{w}C", home="C", away="D", week=w + 1, start=start + timedelta(hours=3),
                            hs=13 if played else None, as_=20 if played else None))
    return gs


def test_freshness_flags_old_snapshot_and_overdue_results():
    gs = _season()
    now = T0 + timedelta(days=13)
    assert live.freshness_problems(gs, now - timedelta(hours=2), now, CFG) == []
    probs = live.freshness_problems(gs, now - timedelta(hours=30), now, CFG)
    assert any("old" in p for p in probs)
    # A week-2 game missing its score after it should be known:
    stale = [replace(g, home_score=None, away_score=None, result_known_at_utc=None) if g.game_id == "W1C" else g
             for g in gs]
    assert any("W1C" in p for p in live.freshness_problems(stale, now - timedelta(hours=2), now, CFG))


def test_targets_respect_lead_and_horizon():
    gs = _season()
    kickoff = T0 + timedelta(days=14)
    assert [g.game_id for g in live.select_targets(gs, kickoff - timedelta(days=1), CFG)] == ["W2A", "W2C"]
    assert [g.game_id for g in live.select_targets(gs, kickoff - timedelta(minutes=5), CFG)] == ["W2C"]
    assert live.select_targets(gs, kickoff - timedelta(days=9), CFG) == []


def test_live_forecast_equals_backtest_forecast_at_same_as_of():
    gs = _season()
    target = next(g for g in gs if g.game_id == "W1C")
    now = target.scheduled_start_utc - timedelta(hours=1)
    # Truncate future results to what a live run would see, then compare with the backtest.
    visible = [g if g.result_known_by(now) else replace(g, home_score=None, away_score=None, result_known_at_utc=None)
               for g in gs]
    [f] = live.forecast_now(visible, EloModel(), now, [target], as_of_lead=timedelta(hours=1))
    bt = {x.game.game_id: x for x in iter_walkforward(gs, EloModel())}
    assert f.as_of == now == bt["W1C"].as_of
    assert f.p_home_win == pytest.approx(bt["W1C"].p_home_win, abs=1e-12)


def test_build_run_refuses_stale_data_and_records_market():
    gs = _season()
    now = T0 + timedelta(days=13)
    recs, run = live.build_run(
        games=gs, factories=[HomeRate, EloModel], now=now, cfg=CFG, as_of_lead=timedelta(hours=1),
        gate=live.GateConfig(), snapshot_id="sha256:abc", manifest={**MANIFEST, "retrieved_at_utc": live._iso(now)},
        market={"W2A": 0.61}, extra={"code_commit": "deadbee"},
    )
    assert {r.game_id for r in recs} == {"W2A", "W2C"} and len(recs) == 4
    assert all(r.data_kind == "real" and not r.simulation_only for r in recs)
    assert all(r.prediction_as_of_utc == live._iso(now) for r in recs)
    assert run["market_p_home_at_issue"] == {"W2A": 0.61} and run["code_commit"] == "deadbee"
    with pytest.raises(live.StaleDataError):
        live.build_run(games=gs, factories=[EloModel], now=now, cfg=CFG, as_of_lead=timedelta(hours=1),
                       gate=live.GateConfig(), snapshot_id="s", manifest=MANIFEST | {"retrieved_at_utc": live._iso(now - timedelta(days=3))},
                       market={}, extra={})


def _rec(gid, model, as_of, p, status="publish"):
    return {"game_id": gid, "model_name": model, "prediction_as_of_utc": live._iso(as_of),
            "distribution": {"p_home_win": p}, "quality_gate": {"status": status}}


def test_scoring_uses_latest_pre_kickoff_forecast_and_excludes_late_ones():
    g1 = make_game(gid="G1", home="A", away="B", start=T0, hs=27, as_=10)
    g2 = make_game(gid="G2", home="C", away="D", start=T0 + timedelta(hours=3), hs=None, as_=None)
    g3 = make_game(gid="G3", home="E", away="F", start=T0 + timedelta(hours=1), hs=10, as_=17)
    games = {g.game_id: g for g in (g1, g2, g3)}
    ledger = [
        _rec("G1", "m", T0 - timedelta(days=2), 0.40),
        _rec("G1", "m", T0 - timedelta(hours=2), 0.70),   # latest before kickoff -> scored
        _rec("G1", "m", T0 + timedelta(minutes=5), 0.99),  # after kickoff -> excluded
        _rec("G2", "m", T0 - timedelta(hours=2), 0.55),   # not final yet -> pending
        _rec("G3", "m", T0 - timedelta(hours=2), 0.30),
        _rec("GX", "m", T0 - timedelta(hours=2), 0.50),   # not in snapshot
    ]
    res = live.score_ledger(ledger, games, {"market_at_issue": {"G1": 0.65}})
    s = res.sets["m"]
    assert dict(zip(s.game_ids, s.p)) == {"G1": 0.70, "G3": 0.30}
    assert res.pending == 1
    assert res.excluded["issued_at_or_after_kickoff"] == 1 and res.excluded["game_not_in_snapshot"] == 1
    assert res.sets["market_at_issue"].game_ids == ["G1"]


def test_market_at_issue_ignores_post_kickoff_runs():
    g = make_game(gid="G1", start=T0)
    t1, t2 = T0 - timedelta(hours=3), T0 + timedelta(minutes=1)
    runs = [{"issued_at_utc": live._iso(t1), "market_p_home_at_issue": {"G1": 0.6}},
            {"issued_at_utc": live._iso(t2), "market_p_home_at_issue": {"G1": 0.9}}]
    ledger = [_rec("G1", "m", t1, 0.5), _rec("G1", "m", t2, 0.5)]
    assert live.market_at_issue(runs, ledger, {"G1": g}) == {"G1": 0.6}


def test_jsonl_round_trip(tmp_path):
    p = tmp_path / "l.jsonl"
    live.append_jsonl(p, [{"a": 1}])
    live.append_jsonl(p, [{"b": 2}])
    assert live.read_jsonl(p) == [{"a": 1}, {"b": 2}]
    assert live.read_jsonl(tmp_path / "missing.jsonl") == []
