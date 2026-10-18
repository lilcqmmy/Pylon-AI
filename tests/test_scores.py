from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from sfi.distributions import DiscreteDist, from_normal, from_residuals, wilson_interval
from sfi.evaluation import scores_eval as SE
from sfi.models.elo import EloParams
from sfi.models.scores import EloScores, NaiveScores, PointsModel, PointsParams

from .conftest import T0, make_game


# ---------------------------------------------------------------- distributions
def test_normal_discretization_moments_and_intervals():
    d = from_normal(3.0, 13.5, -70, 70)
    assert d.mean() == pytest.approx(3.0, abs=0.01) and d.std() == pytest.approx(13.5, abs=0.05)
    for lv in (0.5, 0.8, 0.9):
        a, b = d.central_interval(lv)
        assert d.mass(a, b) >= lv - 1e-9          # discrete intervals are conservative
        assert d.mass(a + 1, b - 1) < lv + 1e-9   # but not wider than needed by more than one step


def test_crps_matches_brute_force_including_outside_support():
    d = from_normal(-2.0, 7.0, -30, 30)
    ks = np.arange(-200, 201)
    F = np.array([d.cdf_at(k) for k in ks])
    for y in (-50, -3, 0, 7, 44):
        assert d.crps(y) == pytest.approx(float(((F - (ks >= y)) ** 2).sum()), abs=1e-9)


def test_point_mass_crps_is_absolute_error():
    d = DiscreteDist(0, np.array([0, 0, 1.0]))  # all mass on 2
    assert d.crps(2) == 0 and d.crps(5) == 3 and d.crps(-1) == 3


def test_residual_shape_keeps_key_numbers():
    res = np.array([-3] * 40 + [3] * 40 + [0] * 20)
    d = from_residuals(0.0, res, -10, 10)
    assert d.prob(3) == pytest.approx(0.4) and d.prob(1) == 0


def test_calibrated_forecasts_cover_and_have_uniform_pit():
    rng = np.random.default_rng(1)
    d = from_normal(0.0, 10.0, -60, 60)
    ys = rng.choice(d.support, size=20000, p=d.pmf)
    lo, hi = d.central_interval(0.8)
    hit = np.mean((ys >= lo) & (ys <= hi))
    assert hit == pytest.approx(d.mass(lo, hi), abs=0.01)
    pits = [d.pit(int(y), rng.uniform()) for y in ys]
    counts = np.histogram(pits, bins=10, range=(0, 1))[0]
    chi2 = ((counts - 2000) ** 2 / 2000).sum()
    assert chi2 < SE.CHI2_DF9_95


def test_wilson_interval():
    lo, hi = wilson_interval(80, 100)
    assert lo < 0.8 < hi and 0.70 < lo < 0.72 and 0.86 < hi < 0.88


# ---------------------------------------------------------------- models
def _league(n_weeks=10, seed=0):
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    off = {"A": 7, "B": 2, "C": -2, "D": -7}
    games = []
    for w in range(n_weeks):
        order = rng.permutation(teams)
        for k, (h, a) in enumerate(((order[0], order[1]), (order[2], order[3]))):
            hs = int(max(0, rng.normal(22 + off[h] + 1.5, 9)))
            as_ = int(max(0, rng.normal(21 + off[a] - 1.5, 9)))
            games.append(make_game(gid=f"G{w}{k}", home=h, away=a, week=w + 1,
                                   start=T0 + timedelta(days=7 * w, hours=3 * k), hs=hs, as_=as_))
    return games


@pytest.mark.parametrize("factory", [
    lambda: NaiveScores(),
    lambda: EloScores(EloParams(), min_samples=5),
    lambda: PointsModel(residual_method="empirical", min_samples=5),
])
def test_score_models_never_see_unreleased_results(factory):
    games = _league()
    cutoff = games[len(games) // 2].scheduled_start_utc
    tampered = [replace(g, home_score=g.home_score + 40) if g.scheduled_start_utc >= cutoff else g for g in games]
    first_known = min(g.result_known_at_utc for g in games if g.scheduled_start_utc >= cutoff)

    def run(gs):
        return {f.game.game_id: (f.as_of, f.margin.mean(), f.total.mean()) for f in SE.iter_score_walkforward(gs, factory())}

    base, after = run(games), run(tampered)
    checked = 0
    for gid, (as_of, m, t) in base.items():
        if as_of < first_known:
            assert after[gid][1:] == (m, t), gid
            checked += 1
    assert checked > 0
    assert any(after[g][1:] != base[g][1:] for g in base if base[g][0] >= first_known)


def test_points_model_learns_strength_and_ignores_home_edge_at_neutral_site():
    games = _league(n_weeks=30, seed=2)
    m = PointsModel(PointsParams(k=0.1), min_samples=5)
    list(SE.iter_score_walkforward(games, m))
    assert m.off["A"] > m.off["D"]
    g = make_game(gid="N", home="A", away="D", start=games[-1].scheduled_start_utc + timedelta(days=7), hs=None, as_=None)
    eh, ea = m.expected_points(g)
    en, an = m.expected_points(replace(g, neutral_site=True))
    assert (eh - ea) - (en - an) == pytest.approx(m.home, abs=1e-9)


def test_score_record_summarizes_distribution_and_rejects_post_kickoff():
    games = _league(2)
    model = NaiveScores()
    fc = next(iter(SE.iter_score_walkforward(games, model)))
    rec = SE.make_score_record(fc, model, source_snapshot_id="sha256:abc", data_kind="synthetic")
    assert rec.outcome_type == "home_margin_and_total"
    assert {"mean", "sd", "q05", "q50", "q95"} <= rec.distribution["margin"].keys()
    assert 0 < rec.distribution["p_home_win_implied"] < 1
    late = replace(fc, as_of=fc.game.scheduled_start_utc)
    with pytest.raises(ValueError):
        SE.make_score_record(late, model, source_snapshot_id="s", data_kind="synthetic")


def test_evaluate_reports_coverage_and_market_point_errors():
    games = _league(12, seed=3)
    fcs = [("points", f) for f in SE.iter_score_walkforward(games, PointsModel(min_samples=5))]
    evals = SE.evaluate(fcs, "margin", {2024})
    s = SE.summarize_target(evals, n_boot=100)["points"]
    assert s["n"] == len(games) and set(s["coverage"]) == {"0.5", "0.8", "0.9"}
    lines = {g.game_id: {"spread": 0.0, "total": 41.0} for g in games}
    mk = SE.market_point_eval("total", lines, {g.game_id: g for g in games}, {2024})
    assert len(mk.y) == len(games) and mk.abs_err_median[0] == abs(mk.y[0] - 41.0)
