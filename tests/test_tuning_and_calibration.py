from dataclasses import replace

import numpy as np
import pytest

from sfi.evaluation.tuning import evaluate_params, grid_params, tune_elo
from sfi.evaluation.walkforward import iter_walkforward, run_walkforward
from sfi.models.calibration import PlattCalibrated
from sfi.models.elo import EloModel, EloParams

from .conftest import make_game

GRID = {"k_factor": [10, 30], "home_field_elo": [0, 50]}
VAL = {2022}


def test_grid_expands_all_combinations_and_rejects_unknown_keys():
    ps = grid_params(EloParams(), GRID)
    assert len(ps) == 4 and len(set(ps)) == 4
    with pytest.raises(ValueError, match="unknown"):
        grid_params(EloParams(), {"learning_rate": [0.1]})


def test_tuning_is_sorted_and_deterministic(small_world):
    games, _ = small_world
    a = tune_elo(games, GRID, VAL)
    b = tune_elo(games, GRID, VAL)
    assert [t.params for t in a] == [t.params for t in b]
    assert [t.brier for t in a] == sorted(t.brier for t in a)
    assert a[0].n > 0


def test_holdout_results_cannot_influence_tuning(small_world):
    games, _ = small_world  # seasons 2021-2023; validation 2022, holdout 2023
    tampered = [replace(g, home_score=g.away_score, away_score=g.home_score) if g.season == 2023 else g
                for g in games]
    assert tune_elo(games, GRID, VAL) == tune_elo(tampered, GRID, VAL)


def test_tuning_scores_only_validation_seasons(small_world):
    games, _ = small_world
    t = evaluate_params(games, EloParams(), {2022})
    expected = sum(1 for g in games if g.season == 2022 and g.home_outcome != 0.5)
    assert t.n == expected


def test_platt_is_identity_until_min_samples(small_world):
    games, _ = small_world
    raw = [f.p_home_win for f in iter_walkforward(games, EloModel(), with_inputs=False)]
    cal = PlattCalibrated(EloModel(), min_samples=10**9)
    wrapped = [f.p_home_win for f in iter_walkforward(games, cal, with_inputs=False)]
    assert wrapped == pytest.approx(raw, abs=1e-9)


def test_platt_learns_only_from_issued_forecasts_of_known_results(small_world):
    games, _ = small_world
    cal = PlattCalibrated(EloModel(), min_samples=1, exclude_first_season=True)
    list(iter_walkforward(games, cal, with_inputs=False))
    first = min(g.season for g in games)
    last_week = max((g.season, g.week) for g in games)
    expected = sum(
        1 for g in games
        if g.season != first and g.home_outcome != 0.5 and (g.season, g.week) != last_week
    )
    assert cal.n_samples == expected  # final week's results are never known before a forecast


def test_platt_shrinks_overconfident_forecasts():
    class Overconfident:
        """Issues probabilities twice as extreme (in logit) as the truth."""
        name, version, feature_spec_version, season = "over", "1", "x", None

        def __init__(self):
            self.rng = np.random.default_rng(5)
            self.truth = {}

        def start_season(self, s): self.season = s
        def prior_games(self, t): return 0
        def inputs_for(self, g): return {}
        def observe(self, g): pass

        def predict_home_win(self, g):
            z = self.rng.normal(0, 1)
            self.truth[g.game_id] = 1 / (1 + np.exp(-z))
            return float(1 / (1 + np.exp(-2 * z)))

    base = Overconfident()
    cal = PlattCalibrated(base, min_samples=50, exclude_first_season=False)
    cal.start_season(2024)
    rng = np.random.default_rng(6)
    for i in range(3000):
        g = make_game(gid=f"G{i}")
        cal.predict_home_win(g)
        win = rng.uniform() < base.truth[g.game_id]
        cal.observe(replace(g, home_score=21 if win else 10, away_score=10 if win else 21))
    cal.predict_home_win(make_game(gid="refit"))
    a, b = cal.coefficients
    assert b == pytest.approx(0.5, abs=0.07) and abs(a) < 0.1


def test_calibrated_records_archive_raw_probability(small_world):
    games, _ = small_world
    recs = run_walkforward(games, [lambda: PlattCalibrated(EloModel(), min_samples=50)],
                           source_snapshot_id="sha256:abc", data_kind="synthetic")
    r = recs[-1]
    assert r.model_name == "baseline_elo_platt" and "+platt" in r.model_version
    assert {"raw_p_home_win", "platt_a", "platt_b", "platt_n"} <= r.inputs.keys()
    assert r.inputs["platt_n"] >= 50
