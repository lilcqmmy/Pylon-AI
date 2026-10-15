import pytest

from sfi.data import synthetic
from sfi.evaluation.report import ORACLE, score, summarize
from sfi.evaluation.walkforward import GateConfig, run_walkforward
from sfi.models.baselines import CoinFlip, HomeRate
from sfi.models.elo import EloModel, EloParams

from .conftest import make_game


def test_elo_equal_teams_favor_home_and_updates_are_zero_sum():
    m = EloModel(EloParams(home_field_elo=48))
    m.start_season(2024)
    g = make_game(hs=30, as_=10)
    assert m.predict_home_win(g) > 0.5
    total = m.ratings["A"] + m.ratings["B"]
    m.observe(g)
    assert m.ratings["A"] > 1500 > m.ratings["B"]
    assert m.ratings["A"] + m.ratings["B"] == pytest.approx(total)


def test_elo_season_regression_pulls_toward_mean():
    m = EloModel(EloParams(season_regression=0.5))
    m.start_season(2023)
    m.ratings["A"] = 1600.0
    m.start_season(2024)
    assert m.ratings["A"] == pytest.approx(1550.0)
    with pytest.raises(ValueError):
        m.start_season(2023)


def test_home_rate_uses_only_observed_games():
    m = HomeRate()
    m.start_season(2024)
    assert m.predict_home_win(make_game()) == 0.5
    m.observe(make_game(hs=20, as_=10))
    assert m.predict_home_win(make_game()) == pytest.approx(2 / 3)


def test_gate_abstains_on_sparse_data(small_world):
    games, _ = small_world
    recs = run_walkforward(games, [EloModel], source_snapshot_id="t", data_kind="synthetic",
                           gate=GateConfig(min_prior_games_per_team=2))
    week1 = [r for r in recs if r.week == 1]
    assert week1 and all(r.quality_gate.status == "abstain" for r in week1)
    assert all("sparse_data" in r.quality_gate.reasons[0] for r in week1)
    assert any(r.quality_gate.status == "publish" for r in recs if r.week >= 3)


def test_end_to_end_elo_beats_coin_flip_and_trails_oracle():
    games, truth = synthetic.generate(20261015, [2021, 2022, 2023, 2024])
    recs = run_walkforward(games, [EloModel, HomeRate, CoinFlip],
                           source_snapshot_id="t", data_kind="synthetic")
    oracle = {t.game_id: t.p_home_win_given_no_tie for t in truth}
    sets = score(recs, {g.game_id: g for g in games}, {2021}, oracle=oracle)
    s = summarize(sets, n_boot=200)
    assert s["coin_flip"]["brier"] == pytest.approx(0.25)
    assert s[ORACLE]["brier"] < s["baseline_elo"]["brier"] < s["coin_flip"]["brier"]
    assert s["baseline_elo"]["n_scored"] == s["coin_flip"]["n_scored"] > 0
