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


# ---------------------------------------------------------------- margin/total in live runs
import numpy as np  # noqa: E402

from sfi.distributions import from_normal  # noqa: E402
from sfi.evaluation import scores_eval as SE  # noqa: E402
from sfi.models.scores import NaiveScores, PointsModel  # noqa: E402


def test_pmf_json_round_trip_is_exact_enough():
    d = from_normal(3.2, 13.0, -70, 70)
    back = SE.pmf_from_json(SE.pmf_to_json(d))
    assert back.quantile(0.1) == d.quantile(0.1) and abs(back.mean() - d.mean()) < 1e-3
    assert abs(back.crps(7) - d.crps(7)) < 1e-3


def test_live_score_forecast_equals_backtest_at_same_as_of():
    gs = _season()
    target = next(g for g in gs if g.game_id == "W1C")
    now = target.scheduled_start_utc - timedelta(hours=1)
    visible = [g if g.result_known_by(now) else replace(g, home_score=None, away_score=None, result_known_at_utc=None)
               for g in gs]
    [f] = live.score_forecast_now(visible, PointsModel(min_samples=1), now, [target], as_of_lead=timedelta(hours=1))
    bt = {x.game.game_id: x for x in SE.iter_score_walkforward(gs, PointsModel(min_samples=1))}
    assert f.as_of == bt["W1C"].as_of == now
    assert np.allclose(f.margin.pmf, bt["W1C"].margin.pmf) and f.margin.lo == bt["W1C"].margin.lo
    assert np.allclose(f.total.pmf, bt["W1C"].total.pmf)


def test_build_run_adds_score_records_with_pmfs_and_lines():
    gs = _season()
    now = T0 + timedelta(days=13)
    recs, run = live.build_run(
        games=gs, factories=[EloModel], now=now, cfg=CFG, as_of_lead=timedelta(hours=1), gate=live.GateConfig(),
        snapshot_id="sha256:abc", manifest={**MANIFEST, "retrieved_at_utc": live._iso(now)}, market={},
        extra={}, score_factories=[NaiveScores, lambda: PointsModel(min_samples=1)],
        lines={"W2A": {"spread": 3.5, "total": 44.0}})
    score_recs = [r for r in recs if r.outcome_type == "home_margin_and_total"]
    assert {r.model_name for r in score_recs} == {"scores_naive", "scores_points_empirical"}
    assert all("margin_pmf" in r.distribution and "total_pmf" in r.distribution for r in score_recs)
    assert run["market_lines_at_issue"] == {"W2A": {"spread": 3.5, "total": 44.0}}


def test_ledger_scoring_separates_win_and_distribution_records():
    g1 = make_game(gid="G1", home="A", away="B", start=T0, hs=27, as_=20)
    g2 = make_game(gid="G2", home="C", away="D", start=T0 + timedelta(hours=3), hs=None, as_=None)
    games = {"G1": g1, "G2": g2}
    t_early, t_late = T0 - timedelta(hours=30), T0 - timedelta(hours=2)
    d_old, d_new = from_normal(0.0, 13.0, -70, 70), from_normal(6.0, 13.0, -70, 70)
    tot = from_normal(45.0, 13.0, 0, 120)

    def srec(gid, t, d):
        return {"game_id": gid, "model_name": "scores_points_empirical", "outcome_type": "home_margin_and_total",
                "prediction_as_of_utc": live._iso(t), "quality_gate": {"status": "publish"},
                "distribution": {"margin_pmf": SE.pmf_to_json(d), "total_pmf": SE.pmf_to_json(tot)}}

    ledger = [_rec("G1", "elo_tuned", t_late, 0.6), srec("G1", t_early, d_old), srec("G1", t_late, d_new),
              srec("G2", t_late, d_new)]
    runs = [{"issued_at_utc": live._iso(t_late), "market_lines_at_issue": {"G1": {"spread": 4.0, "total": 47.0}}}]
    win = live.score_ledger(ledger, games)
    assert win.sets["elo_tuned"].p == [0.6]  # distribution records are ignored here
    res = live.score_ledger_distributions(ledger, runs, games, {"G1": {"spread": 5.0, "total": 46.0}})
    e = res.evals["margin"]["scores_points_empirical"]
    assert e.game_ids == ["G1"] and e.crps[0] == pytest.approx(SE.pmf_from_json(SE.pmf_to_json(d_new)).crps(7), abs=1e-9)
    assert res.pending == 1
    assert res.market["margin"]["at_issue"].abs_err_median == [3.0]
    assert res.market["total"]["snapshot"].abs_err_median == [1.0]


# ---------------------------------------------------------------- player yards in live runs
from sfi.evaluation import players_eval as PE  # noqa: E402
from sfi.models.players import PlayerOppEff, PlayerParams, PlayerRecentAvg  # noqa: E402


def _player_stats(gs):
    st = {}
    for g in gs:
        if not g.is_final:
            continue
        st[g.game_id] = [("q" + g.home_team, g.home_team, "pass", 30.0, 240.0),
                         ("r" + g.home_team, g.home_team, "rush", 15.0, 60.0),
                         ("q" + g.away_team, g.away_team, "pass", 28.0, 210.0),
                         ("w" + g.away_team, g.away_team, "rec", 7.0, 55.0)]
    return st


def test_p_at_least_has_no_push_ambiguity():
    d = from_normal(250.0, 60.0, -20, 600)
    assert PE.p_at_least(d, 250) == pytest.approx(1 - d.cdf_at(249))
    assert PE.p_at_least(d, 200) > PE.p_at_least(d, 250) > PE.p_at_least(d, 300)


def test_live_player_forecast_equals_backtest_at_same_as_of():
    gs = _season()
    st = _player_stats(gs)
    target = next(g for g in gs if g.game_id == "W1C")
    now = target.scheduled_start_utc - timedelta(hours=1)
    visible = [g if g.result_known_by(now) else replace(g, home_score=None, away_score=None, result_known_at_utc=None)
               for g in gs]
    live_fc = {f.player + f.stat: f for _, f in live.player_forecast_now(
        visible, PlayerOppEff(st, params=PlayerParams(min_k=1)), now, [target], as_of_lead=timedelta(hours=1))}
    bt = {f.player + f.stat: f for a, f in PE.iter_player_walkforward(gs, PlayerOppEff(st, params=PlayerParams(min_k=1)))
          if f.game.game_id == "W1C"}
    assert live_fc.keys() == bt.keys() and live_fc
    for k in bt:
        assert live_fc[k].mu == pytest.approx(bt[k].mu) and np.allclose(live_fc[k].dist.pmf, bt[k].dist.pmf)


def _run_with_players(gs, st, now):
    return live.build_run(
        games=gs, factories=[EloModel], now=now, cfg=CFG, as_of_lead=timedelta(hours=1), gate=live.GateConfig(),
        snapshot_id="sha256:abc", manifest={**MANIFEST, "retrieved_at_utc": live._iso(now)}, market={}, extra={},
        player_factory=lambda: PlayerRecentAvg(st, params=PlayerParams(min_k=1)), player_stats=st,
        thresholds={"pass": [200, 250], "rush": [50], "rec": [50]})


def test_build_run_adds_player_records_with_pmf_and_thresholds():
    gs = _season()
    now = T0 + timedelta(days=13)
    recs, run = _run_with_players(gs, _player_stats(gs), now)
    pr = [r for r in recs if r.outcome_type.startswith("player_")]
    assert pr and all("pmf" in r.distribution for r in pr)
    qb = next(r for r in pr if r.outcome_type == "player_pass_yards")
    assert set(qb.distribution["p_at_least"]) == {"200", "250"}
    assert run["players"] == {"skipped": False, "recent_pbp_gaps": []}


def test_players_skipped_when_play_by_play_lags_too_long():
    gs = _season()
    st = _player_stats(gs)
    del st[gs[0].game_id]  # a week-1 game is final but has no play-by-play
    now = T0 + timedelta(days=13)
    recs, run = _run_with_players(gs, st, now)
    assert not any(r.outcome_type.startswith("player_") for r in recs)
    assert run["players"]["skipped"] and run["players"]["games"] == [gs[0].game_id]
    # A gap younger than the limit is recorded but does not block forecasts
    gs2 = _season()
    st2 = _player_stats(gs2)
    last_final = max((g for g in gs2 if g.is_final), key=lambda g: g.scheduled_start_utc)
    del st2[last_final.game_id]
    now2 = last_final.result_known_at_utc + timedelta(hours=2)
    recs2, run2 = _run_with_players(gs2, st2, now2)
    assert run2["players"] == {"skipped": False, "recent_pbp_gaps": [last_final.game_id]}


def test_player_ledger_scoring_latest_zero_and_pending():
    g1 = make_game(gid="G1", home="A", away="B", start=T0, hs=27, as_=20)
    g2 = make_game(gid="G2", home="C", away="D", start=T0 + timedelta(hours=3), hs=24, as_=21)  # final, no pbp yet
    games = {"G1": g1, "G2": g2}
    d_old, d_new = from_normal(150.0, 50.0, -20, 600), from_normal(260.0, 60.0, -20, 600)

    def prec(gid, player, stat, t, d):
        return {"game_id": gid, "model_name": "players_opp_eff", "outcome_type": f"player_{stat}_yards",
                "prediction_as_of_utc": live._iso(t),
                "distribution": {"player": player, "pmf": SE.pmf_to_json(d),
                                 "p_at_least": {"250": round(PE.p_at_least(d, 250), 6)}}}

    t1, t2 = T0 - timedelta(days=2), T0 - timedelta(hours=2)
    ledger = [prec("G1", "qa", "pass", t1, d_old), prec("G1", "qa", "pass", t2, d_new),
              prec("G1", "wa", "rec", t2, from_normal(50.0, 25.0, -10, 300)),
              prec("G2", "qc", "pass", t2, d_new), _rec("G1", "elo_tuned", t2, 0.6)]
    stats = {"G1": [("qa", "A", "pass", 33.0, 281.0)]}  # wa had no target -> 0 yards
    res = live.score_ledger_players(ledger, games, stats, "players_opp_eff")
    e = res.evals["pass"]
    assert e.y == [281] and e.crps[0] == pytest.approx(SE.pmf_from_json(SE.pmf_to_json(d_new)).crps(281))
    assert res.evals["rec"].y == [0] and res.no_opportunity == {"rec": 1}
    assert res.pending == 1
    assert res.thresholds["pass"]["250"][0][1] == 1.0


def test_market_moves_use_issuing_run_and_closing_snapshot():
    g1 = make_game(gid="G1", home="A", away="B", start=T0, hs=24, as_=10)
    g2 = make_game(gid="G2", home="C", away="D", start=T0, hs=None, as_=None)  # not final: skipped
    games = {"G1": g1, "G2": g2}
    t_early, t_late, t_after = T0 - timedelta(days=2), T0 - timedelta(hours=3), T0 + timedelta(minutes=1)
    ledger = [_rec("G1", "elo_tuned", t_early, 0.55), _rec("G1", "elo_tuned", t_late, 0.58),
              _rec("G1", "elo_tuned", t_after, 0.99), _rec("G2", "elo_tuned", t_late, 0.5)]
    runs = [{"issued_at_utc": live._iso(t_early), "market_p_home_at_issue": {"G1": 0.50}},
            {"issued_at_utc": live._iso(t_late), "market_p_home_at_issue": {"G1": 0.53, "G2": 0.5}},
            {"issued_at_utc": live._iso(t_after), "market_p_home_at_issue": {"G1": 0.90}}]
    [m] = live.market_moves(ledger, runs, games, {"G1": 0.60, "G2": 0.5}, "sha256:x")
    assert m["p_home_at_issue"] == 0.53 and m["p_home_close"] == 0.60
    assert m["market_move"] == pytest.approx(0.07) and m["model_p_home"] == 0.58 and m["home_outcome"] == 1.0
