"""Point-in-time guarantees: forecasts may depend only on results known before the as-of time."""

from dataclasses import replace
from datetime import timedelta

import pytest

from sfi.data.schema import known_results
from sfi.evaluation.walkforward import run_walkforward
from sfi.forecasts.record import QualityGate, make_record
from sfi.models.elo import EloModel

from .conftest import T0, make_game


def _forecasts(games):
    recs = run_walkforward(games, [EloModel], source_snapshot_id="t", data_kind="synthetic")
    return {r.game_id: r.distribution["p_home_win"] for r in recs}


def test_future_results_do_not_change_past_forecasts(small_world):
    games, _ = small_world
    cutoff = (2022, 5)
    base = _forecasts(games)
    # Flip every score from the cutoff week onward.
    tampered = [
        replace(g, home_score=g.away_score, away_score=g.home_score) if (g.season, g.week) >= cutoff else g
        for g in games
    ]
    after = _forecasts(tampered)
    for g in games:
        if (g.season, g.week) <= cutoff:  # the cutoff week itself is forecast before its own results
            assert after[g.game_id] == base[g.game_id], g.game_id
    assert any(after[g.game_id] != base[g.game_id] for g in games if (g.season, g.week) > cutoff)


class SpyElo(EloModel):
    seen: list = []

    def observe(self, game):
        SpyElo.seen.append(game)
        super().observe(game)


def test_model_only_observes_results_known_before_as_of(small_world):
    games, _ = small_world
    SpyElo.seen = []
    recs = run_walkforward(games, [SpyElo], source_snapshot_id="t", data_kind="synthetic")
    as_of_by_week = {(r.season, r.week): r.prediction_as_of_utc for r in recs}
    # Every result observed must have been known by the as-of of the next week it could influence.
    seen_ids = [g.game_id for g in SpyElo.seen]
    assert len(seen_ids) == len(set(seen_ids)), "a result was observed twice"
    for g in SpyElo.seen:
        later = [k for k in as_of_by_week if k > (g.season, g.week)]
        assert later, "results of the final week should never be observed"
        first_after = min(later)
        assert g.result_known_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ") <= as_of_by_week[first_after]


def test_late_correction_is_invisible_before_it_is_known():
    g = make_game(known=T0 + timedelta(days=3))
    assert known_results([g], T0 + timedelta(days=2)) == []
    assert known_results([g], T0 + timedelta(days=3)) == [g]


def test_forecast_must_be_made_before_kickoff():
    g = make_game()
    with pytest.raises(ValueError, match="not before kickoff"):
        make_record(game=g, as_of=T0, p_home_win=0.6, model_name="m", model_version="1",
                    feature_spec_version="f", source_snapshot_id="s", inputs={},
                    gate=QualityGate("publish"), data_kind="synthetic")
