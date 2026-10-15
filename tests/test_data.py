from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from sfi.data import synthetic
from sfi.data.schema import ValidationError, file_sha256, read_games_csv, validate_games, write_games_csv

from .conftest import T0, make_game


def test_validation_rejects_bad_records():
    g = make_game()
    with pytest.raises(ValidationError, match="duplicate"):
        validate_games([g, g])
    with pytest.raises(ValidationError, match="before kickoff"):
        validate_games([make_game(known=T0 - timedelta(minutes=1))])
    with pytest.raises(ValidationError, match="twice in a week"):
        validate_games([g, make_game(gid="G2", home="A", away="C")])
    with pytest.raises(ValidationError, match="plays itself"):
        validate_games([make_game(home="A", away="A")])


def test_csv_round_trip_preserves_games(tmp_path, small_world):
    games, _ = small_world
    p = tmp_path / "g.csv"
    write_games_csv(games, p)
    assert read_games_csv(p) == games
    write_games_csv(games, tmp_path / "g2.csv")
    assert file_sha256(p) == file_sha256(tmp_path / "g2.csv")


def test_unplayed_games_round_trip(tmp_path):
    g = make_game(hs=None, as_=None)
    write_games_csv([g], tmp_path / "u.csv")
    assert read_games_csv(tmp_path / "u.csv") == [g]


def test_generator_is_deterministic_and_valid():
    a, ta = synthetic.generate(3, [2020, 2021], 8, 5)
    b, tb = synthetic.generate(3, [2020, 2021], 8, 5)
    assert a == b and ta == tb
    validate_games(a)
    assert len(a) == 2 * 5 * 4


def test_points_pmf_is_a_distribution_with_the_right_mean():
    pmf = synthetic.points_pmf(22.0)
    assert pmf.sum() == pytest.approx(1.0)
    assert float(np.arange(pmf.size) @ pmf) == pytest.approx(22.0, abs=0.05)


def test_truth_matches_simulated_frequencies():
    """Exact outcome probabilities agree with Monte Carlo draws from the same scoring model."""
    hm, am = synthetic.expected_points(3.0, 0.0)
    p_win, p_tie = synthetic.outcome_probs(hm, am)
    rng = np.random.default_rng(11)
    n = 200_000
    h = 7 * rng.poisson(hm * 0.7 / 7, n) + 3 * rng.poisson(hm * 0.3 / 3, n)
    a = 7 * rng.poisson(am * 0.7 / 7, n) + 3 * rng.poisson(am * 0.3 / 3, n)
    assert np.mean(h > a) == pytest.approx(p_win, abs=0.005)
    assert np.mean(h == a) == pytest.approx(p_tie, abs=0.003)
