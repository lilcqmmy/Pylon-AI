from datetime import datetime, timedelta, timezone

import pytest

from sfi.evaluation import players_eval as PE
from sfi.evaluation.walkforward import Replay
from sfi.features.players import aggregate_players_season
from sfi.features.qb import InjuryRow
from sfi.models.players import PlayerOppEff, PlayerParams, PlayerRecentAvg

from .conftest import make_game

T0 = datetime(2015, 9, 13, 17, tzinfo=timezone.utc)


def _games(n=5):
    return [make_game(gid=f"2015_{w + 1:02d}_B_A", home="A", away="B", season=2015, week=w + 1,
                      start=T0 + timedelta(days=7 * w), hs=24, as_=17) for w in range(n)]


def _stats(games, *, backup_week=None, fringe_week=None):
    """A: QB 'qa' (30 att), RB 'ra' (15 car), WR 'wa' (8 tgt). B: QB 'qb'.
    Optional: backup QB 'qa2' throws 12 passes in one week; fringe WR 'fa' gets 1 target in one week."""
    st = {}
    for i, g in enumerate(games):
        rows = [("qa", "A", "pass", 30.0, 250.0), ("ra", "A", "rush", 15.0, 70.0), ("wa", "A", "rec", 8.0, 60.0),
                ("qb", "B", "pass", 32.0, 230.0)]
        if i == backup_week:
            rows.append(("qa2", "A", "pass", 12.0, 80.0))
        if i == fringe_week:
            rows.append(("fa", "A", "rec", 1.0, 5.0))
        st[g.game_id] = rows
    return st


def _forecast(model, games, idx, as_of=None):
    r = Replay(games, model)
    r.advance_to(as_of or games[idx].scheduled_start_utc - timedelta(hours=1))
    r.prepare(games[idx])
    return {(f.team, f.player, f.stat): f for f in model.predict_players(games[idx])}


def test_pool_threshold_and_single_expected_qb():
    games = _games()
    fc = _forecast(PlayerRecentAvg(_stats(games, backup_week=2, fringe_week=2)), games, 3)
    assert ("A", "qa", "pass") in fc and ("A", "qa2", "pass") not in fc   # one expected starter
    assert ("A", "fa", "rec") not in fc                                  # 1 target < min 2 over last 2 games
    assert {("A", "ra", "rush"), ("A", "wa", "rec"), ("B", "qb", "pass")} <= fc.keys()


def test_backup_becomes_expected_starter_when_starter_listed_out_in_time():
    games = _games()
    target = games[3]
    ko = target.scheduled_start_utc
    inj = {(2015, target.week, "A"): [InjuryRow("qa", "Out", ko - timedelta(hours=40), "date_modified")]}
    m = PlayerRecentAvg(_stats(games, backup_week=2), inj)
    early = _forecast(m, games, 3, as_of=ko - timedelta(hours=50))
    assert ("A", "qa", "pass") in early                 # report not known yet
    m2 = PlayerRecentAvg(_stats(games, backup_week=2), inj)
    late = _forecast(m2, games, 3)
    assert ("A", "qa", "pass") not in late and ("A", "qa2", "pass") in late


def test_traded_player_leaves_old_team_pool():
    games = _games()
    st = _stats(games)
    # 'wa' appears for B in week 3 (index 2): latest appearance is for B, so not in A's pool for week 4
    st[games[2].game_id] = [r for r in st[games[2].game_id] if r[0] != "wa"] + [("wa", "B", "rec", 5.0, 40.0)]
    fc = _forecast(PlayerRecentAvg(st), games, 3)
    assert ("A", "wa", "rec") not in fc and ("B", "wa", "rec") in fc


def test_player_forecasts_never_read_unreleased_stats():
    games = _games(6)
    base_stats = _stats(games)
    tampered = {**base_stats, games[3].game_id: [(p, t, s, n, y + 500) for p, t, s, n, y in base_stats[games[3].game_id]]}

    def run(st):
        m = PlayerOppEff(st, params=PlayerParams(min_k=1))
        return {(a, f.game.game_id, f.player, f.stat): f.mu for a, f in PE.iter_player_walkforward(games, m)}

    a, b = run(base_stats), run(tampered)
    for (as_of, gid, p, s), mu in a.items():
        if as_of < games[3].result_known_at_utc:
            assert b[(as_of, gid, p, s)] == mu
    assert any(b[k] != v for k, v in a.items() if k[0] >= games[3].result_known_at_utc)


def test_pool_player_without_opportunity_scores_zero():
    games = _games(4)
    st = _stats(games)
    st[games[3].game_id] = [r for r in st[games[3].game_id] if r[0] != "ra"]  # RB inactive in week 4
    m = PlayerRecentAvg(st)
    fcs = [("m", a, f) for a, f in PE.iter_player_walkforward(games, m)]
    evals, extra = PE.evaluate_players(fcs, st, {2015})
    e = evals[("m", "rush")]
    i = e.game_ids.index(f"{games[3].game_id}|ra|rush")
    assert e.y[i] == 0 and extra[("m", "rush")]["no_opportunity"] == 1


def test_spread_falls_back_until_enough_residuals_then_uses_them():
    m = PlayerRecentAvg({}, params=PlayerParams(min_k=50))
    d = m.spread("rec", 60.0)
    from sfi.distributions import from_normal
    assert d.std() == pytest.approx(from_normal(60.0, 36.0, -10, 300).std())  # Normal fallback, sd = 0.6 * mu
    for _ in range(60):
        m.res_bins[("rec", 6)].append(-60.0)          # all past forecasts near 60 yards ended at 0
    assert m.spread("rec", 60.0).prob(0) == pytest.approx(1.0)


def test_aggregation_definitions(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    rows = [  # passer, rusher, receiver, pass_yds, rush_yds, rec_yds, complete, incomplete, int, two_pt
        ("q", None, "w", 20, None, 20, 1, 0, 0, 0),     # completion
        ("q", None, "w", None, None, None, 0, 1, 0, 0),  # incompletion: attempt + target, 0 yards
        ("q", None, "t", None, None, None, 0, 0, 1, 0),  # interception: attempt + target
        ("q", None, None, None, None, None, 0, 0, 0, 0),  # sack: not an attempt
        (None, "q", None, None, 9, None, 0, 0, 0, 0),    # scramble: carry
        (None, "r", None, None, 4, None, 0, 0, 0, 0),
        ("q", None, "w", 2, None, 2, 1, 0, 0, 1),        # two-point try: excluded
    ]
    names = ["passer_player_id", "rusher_player_id", "receiver_player_id", "passing_yards", "rushing_yards",
             "receiving_yards", "complete_pass", "incomplete_pass", "interception", "two_point_attempt"]
    data = {n: [r[i] for r in rows] for i, n in enumerate(names)}
    for n in names[3:]:
        data[n] = pa.array([None if v is None else float(v) for v in data[n]], type=pa.float64())
    data.update(game_id=["G"] * len(rows), posteam=["A"] * len(rows))
    path = tmp_path / "p.parquet"
    pq.write_table(pa.table(data), path)
    got = {(p, s): (n, y) for p, _, s, n, y in aggregate_players_season(path)["G"]}
    assert got[("q", "pass")] == (3.0, 20.0)
    assert got[("q", "rush")] == (1.0, 9.0) and got[("r", "rush")] == (1.0, 4.0)
    assert got[("w", "rec")] == (2.0, 20.0) and got[("t", "rec")] == (1.0, 0.0)


def test_player_record_is_pregame_and_identifies_player():
    games = _games(3)
    m = PlayerRecentAvg(_stats(games))
    pairs = list(PE.iter_player_walkforward(games, m))
    as_of, f = pairs[-1]
    rec = PE.make_player_record(as_of, f, m, source_snapshot_id="sha256:x", data_kind="real")
    assert rec.outcome_type == f"player_{f.stat}_yards" and rec.distribution["player"] == f.player
    with pytest.raises(ValueError):
        PE.make_player_record(f.game.scheduled_start_utc, f, m, source_snapshot_id="s", data_kind="real")
