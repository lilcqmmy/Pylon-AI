import math

import numpy as np
import pytest

from sfi.evaluation import metrics as M


def test_brier_and_log_loss_known_values():
    p, y = [0.8, 0.3], [1, 0]
    assert M.brier(p, y) == pytest.approx((0.04 + 0.09) / 2)
    assert M.log_loss(p, y) == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2)


def test_coin_flip_reference_values():
    y = [1, 0, 1, 1]
    assert M.brier([0.5] * 4, y) == pytest.approx(0.25)
    assert M.log_loss([0.5] * 4, y) == pytest.approx(math.log(2))


def test_ties_must_be_removed_before_scoring():
    with pytest.raises(ValueError, match="binary"):
        M.brier([0.5], [0.5])


def test_calibrated_forecasts_have_slope_near_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    a, b = M.calibration_intercept_slope(p, y)
    assert abs(a) < 0.05 and abs(b - 1) < 0.05


def test_overconfident_forecasts_have_slope_below_one():
    rng = np.random.default_rng(1)
    true = rng.uniform(0.2, 0.8, 20000)
    y = (rng.uniform(size=true.size) < true).astype(float)
    logit = np.log(true / (1 - true)) * 2.0  # twice as extreme as warranted
    _, b = M.calibration_intercept_slope(1 / (1 + np.exp(-logit)), y)
    assert b == pytest.approx(0.5, abs=0.05)


def test_calibration_slope_is_undefined_for_constant_forecasts():
    a, b = M.calibration_intercept_slope([0.5] * 10, [1, 0] * 5)
    assert math.isnan(a) and math.isnan(b)


def test_calibration_table_counts_everything():
    p = np.linspace(0.01, 0.99, 50)
    y = (p > 0.5).astype(float)
    assert sum(b.n for b in M.calibration_table(p, y, 10)) == 50


def test_block_bootstrap_ci_brackets_mean():
    vals = np.arange(100, dtype=float)
    blocks = np.repeat(np.arange(10), 10)
    m, lo, hi = M.block_bootstrap_mean_ci(vals, blocks, n_boot=500)
    assert lo < m < hi and m == pytest.approx(49.5)
