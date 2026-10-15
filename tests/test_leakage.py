"""Point-in-time guarantees: forecasts may depend only on results known before the as-of time."""

from dataclasses import replace
from datetime import timedelta

import pytest

from sfi.data.schema import known_results
from sfi.evaluation.walkforward import iter_walkforward, run_walkforward
from sfi.forecasts.record import QualityGate, make_record
from sfi.models.elo import EloModel

from .conftest import T0, make_game


def _forecasts(games):
    return {f.game.game_id: (f.as_of, f.p_home_win) for f in iter_walkforward(games, EloModel())}


def test_future_results_do_not_change_past_forecasts(small_world):
    games, _ = small_world
    cutoff = (2022, 5)
    tampered_ids = {g.game_id for g in games if (g.season, g.week) >= cutoff}
    first_known = min(g.result_known_at_utc for g in games if g.game_id in tampered_ids)
    base = _forecasts(games)
    after = _forecasts([
        replace(g, home_score=g.away_score, away_score=g.home_score) if g.game_id in tampered_ids else g
        for g in games
    ])
    n_checked = 0
    for gid, (as_of, p) in base.items():
        if as_of < first_known:
            assert after[gid][1] == p, gid
            n_checked += 1
    assert n_checked > 0
    # Later slates in the cutoff week can see earlier tampered results, so some forecasts change.
    assert any(after[gid][1] != p for gid, (as_of, p) in base.items() if as_of >= first_known)


class SpyElo(EloModel):
    def __init__(self):
        super().__init__()
        self.seen = []

    def observe(self, game):
        self.seen.append(game)
        super().observe(game)


def test_model_only_observes_results_known_before_as_of(small_world):
    games, _ = small_world
    spy = SpyElo()
    n = 0
    for f in iter_walkforward(games, spy):
        # At the moment of each forecast, everything the model has learned from was known by as_of.
        assert all(g.result_known_at_utc <= f.as_of for g in spy.seen)
        assert f.as_of < f.game.scheduled_start_utc
        n += 1
    ids = [g.game_id for g in spy.seen]
    assert n == len(games) and len(ids) == len(set(ids))


def test_later_kickoff_same_day_can_use_earlier_known_result():
    early = make_game(gid="E", home="A", away="B", start=T0)  # known at T0+4h
    late = make_game(gid="L", home="C", away="D", start=T0 + timedelta(hours=6), hs=None, as_=None)
    too_soon = make_game(gid="S", home="E", away="F", start=T0 + timedelta(hours=4), hs=None, as_=None)
    spy = SpyElo()
    seen_at = {f.game.game_id: [g.game_id for g in spy.seen] for f in iter_walkforward([early, late, too_soon], spy)}
    assert seen_at["S"] == []      # as_of T0+3h: result not yet known
    assert seen_at["L"] == ["E"]   # as_of T0+5h: known at T0+4h


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


def test_run_walkforward_records_match_iterator(small_world):
    games, _ = small_world
    recs = run_walkforward(games, [EloModel], source_snapshot_id="t", data_kind="synthetic")
    it = _forecasts(games)
    assert all(abs(r.distribution["p_home_win"] - round(it[r.game_id][1], 6)) < 1e-12 for r in recs)
