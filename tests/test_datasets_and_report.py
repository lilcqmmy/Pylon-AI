import pytest

from sfi.datasets import DatasetError, Splits, parse_seasons
from sfi.evaluation.report import ScoredSet, paired_brier_diff


def test_parse_seasons_forms():
    assert parse_seasons("2001-2003") == {2001, 2002, 2003}
    assert parse_seasons(2000) == {2000}
    assert parse_seasons(["2000", "2002-2003", 2005]) == {2000, 2002, 2003, 2005}
    with pytest.raises(ValueError):
        parse_seasons("2005-2001")


def test_splits_must_be_ordered_disjoint_and_present():
    Splits.from_config({"burn_in": 2000, "validation": "2001-2019", "holdout": "2020-2025"})
    with pytest.raises(DatasetError, match="ordered"):
        Splits.from_config({"burn_in": 2000, "validation": "2010-2019", "holdout": "2005-2008"})
    with pytest.raises(DatasetError, match="disjoint"):
        Splits.from_config({"validation": "2001-2019", "holdout": "2019-2025"})
    sp = Splits.from_config({"validation": 2001, "holdout": 2002})
    with pytest.raises(DatasetError, match="not in the data"):
        sp.check({2001})


def test_paired_comparison_uses_only_shared_games():
    a = ScoredSet("a", ["g1", "g2", "g3"], [0.9, 0.2, 0.6], [1, 0, 1], ["w1", "w1", "w2"])
    b = ScoredSet("b", ["g3", "g1"], [0.5, 0.5], [1, 1], ["w2", "w1"])
    m, lo, hi, n = paired_brier_diff({"a": a, "b": b}, "a", "b", n_boot=200)
    assert n == 2
    assert m == pytest.approx(((0.1**2 - 0.25) + (0.4**2 - 0.25)) / 2)
