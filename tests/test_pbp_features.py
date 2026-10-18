import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sfi.data.sources import nflverse_pbp as P
from sfi.evaluation.walkforward import iter_walkforward
from sfi.features.pbp import SUM_FIELDS, TeamHistory, aggregate_season
from sfi.models.elo import EloParams
from sfi.models.pbp_logit import PbpLogit, fit_logistic_l2

from .conftest import make_game


def sums(plays=60.0, epa=0.0, **kw):
    d = dict.fromkeys(SUM_FIELDS, 0.0)
    d.update(plays=plays, epa=epa, success=plays * 0.45, pass_plays=plays / 2, rush_plays=plays / 2,
             early=plays * 0.7, early_pass=plays * 0.35, dropbacks=plays / 2)
    d.update(kw)
    return d


def test_team_history_rolling_window_and_season_fallback():
    h = TeamHistory(metrics=["epa_play"])
    assert math.isnan(h.features("A")["off_epa_play_r4"])
    for e in (6.0, 0.0, 0.0, 0.0, 0.0):  # five games of 60 plays; the first drops out of the window
        h.observe_game("A", sums(epa=e), sums())
    f = h.features("A")
    assert f["off_epa_play_r4"] == pytest.approx(0.0)
    assert f["off_epa_play_std"] == pytest.approx(6.0 / 300)
    h.start_season()
    f = h.features("A")
    assert f["off_epa_play_std"] == pytest.approx(6.0 / 300)  # falls back to last season's totals
    assert f["off_epa_play_r4"] == pytest.approx(0.0)          # rolling window crosses seasons
    h.observe_game("A", sums(epa=12.0), sums())
    assert h.features("A")["off_epa_play_std"] == pytest.approx(12.0 / 60)


def _games_and_stats(n_weeks=12):
    teams = ["A", "B", "C", "D"]
    rng = np.random.default_rng(3)
    strength = {"A": 0.15, "B": 0.05, "C": -0.05, "D": -0.15}
    games, stats = [], {}
    t0 = datetime(2010, 9, 12, 17, tzinfo=timezone.utc)
    for season_off, season in enumerate((2010, 2011)):
        for w in range(n_weeks):
            start = t0 + timedelta(days=365 * season_off + 7 * w)
            order = rng.permutation(teams)
            for k, (h, a) in enumerate(((order[0], order[1]), (order[2], order[3]))):
                gid = f"{season}_{w:02d}_{a}_{h}"
                win = rng.uniform() < 0.5 + strength[h] - strength[a] + 0.05
                games.append(make_game(gid=gid, home=h, away=a, start=start + timedelta(hours=3 * k),
                                       season=season, week=w + 1, hs=24 if win else 13, as_=13 if win else 24))
                stats[(gid, h)] = sums(epa=60 * (strength[h] + rng.normal(0, 0.1)))
                stats[(gid, a)] = sums(epa=60 * (strength[a] + rng.normal(0, 0.1)))
    return games, stats


def test_pbp_logit_never_reads_stats_of_unreleased_games():
    games, stats = _games_and_stats()
    params = EloParams()

    def run(st):
        m = PbpLogit(params, st, metrics=["epa_play"], min_samples=20)
        return {f.game.game_id: (f.as_of, f.p_home_win) for f in iter_walkforward(games, m)}

    base = run(stats)
    cutoff = games[len(games) // 2].scheduled_start_utc
    tampered = {k: (sums(epa=999.0) if next(g for g in games if g.game_id == k[0]).scheduled_start_utc >= cutoff else v)
                for k, v in stats.items()}
    after = run(tampered)
    earliest_known = min(g.result_known_at_utc for g in games if g.scheduled_start_utc >= cutoff)
    checked = 0
    for gid, (as_of, p) in base.items():
        if as_of < earliest_known:
            assert after[gid][1] == p, gid
            checked += 1
    assert checked > 0
    assert any(after[g][1] != base[g][1] for g in base if base[g][0] >= earliest_known)


def test_pbp_logit_uses_elo_until_enough_samples_then_fits():
    games, stats = _games_and_stats()
    m = PbpLogit(EloParams(), stats, metrics=["epa_play"], min_samples=10**6)
    from sfi.models.elo import EloModel
    e = EloModel(EloParams())
    a = [f.p_home_win for f in iter_walkforward(games, m, with_inputs=False)]
    b = [f.p_home_win for f in iter_walkforward(games, e, with_inputs=False)]
    assert a == pytest.approx(b, abs=1e-12)
    fitted = PbpLogit(EloParams(), stats, metrics=["epa_play"], min_samples=20)
    recs = list(iter_walkforward(games, fitted))
    assert fitted.fit_season == 2011 and recs[-1].inputs["n_train"] > 20
    assert "d_off_epa_play_r4" in recs[-1].inputs["features"]


def test_version_changes_with_data_tag():
    a = PbpLogit(EloParams(), {}, data_tag="sha256:aaa").version
    b = PbpLogit(EloParams(), {}, data_tag="sha256:bbb").version
    assert a != b


def test_ridge_logistic_recovers_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 2))
    y = (rng.uniform(size=5000) < 1 / (1 + np.exp(-(0.3 + 1.2 * X[:, 0])))).astype(float)
    beta = fit_logistic_l2(np.column_stack([np.ones(5000), X]), y, l2=1.0)
    assert beta[0] == pytest.approx(0.3, abs=0.1) and beta[1] == pytest.approx(1.2, abs=0.1)
    assert abs(beta[2]) < 0.1


def test_aggregate_season_counts_and_exclusions(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    rows = [  # (posteam, pass, rush, epa, yards, down, int, fum, sack, hit, dropback, two_pt, spike, kneel)
        ("A", 1, 0, 0.5, 25, 1, 0, 0, 0, 1, 1, 0, 0, 0),   # explosive pass, early, pressured
        ("A", 0, 1, -0.2, 3, 2, 0, 1, 0, 0, 0, 0, 0, 0),   # rush, early, fumble lost
        ("A", 1, 0, -1.0, -7, 3, 0, 0, 1, 0, 1, 0, 0, 0),  # sack
        ("A", 1, 0, 0.9, 2, None, 0, 0, 0, 0, 1, 1, 0, 0),  # two-point try: excluded
        ("A", 0, 1, -0.1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1),    # kneel: excluded
        ("A", 0, 0, 0.3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0),     # punt etc.: excluded
        ("B", 0, 1, None, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0),    # missing EPA: excluded
    ]
    cols = ["posteam", "pass", "rush", "epa", "yards_gained", "down", "interception", "fumble_lost",
            "sack", "qb_hit", "qb_dropback", "two_point_attempt", "qb_spike", "qb_kneel"]
    data = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    data.update(game_id=["G"] * len(rows), defteam=["B" if r[0] == "A" else "A" for r in rows], success=[1, 0, 0, 1, 0, 0, 1])
    for c in cols[1:]:
        data[c] = pa.array([None if v is None else float(v) for v in data[c]], type=pa.float64())
    path = tmp_path / "p.parquet"
    pq.write_table(pa.table(data), path)
    out = aggregate_season(path)
    a = out[("G", "A")]
    assert a["plays"] == 3 and a["pass_plays"] == 2 and a["rush_plays"] == 1
    assert a["epa"] == pytest.approx(-0.7) and a["explosive"] == 1 and a["early"] == 2 and a["early_pass"] == 1
    assert a["turnovers"] == 1 and a["dropbacks"] == 2 and a["pressure"] == 2
    assert ("G", "B") not in out


def test_fetch_pbp_skips_unchanged_assets(tmp_path):
    body = {"play_by_play_2026.parquet": b"x" * 10}
    meta = {"assets": [{"name": "play_by_play_2026.parquet", "size": 10, "updated_at": "2026-10-18T13:12:43Z"}]}
    calls = []

    def get(url):
        calls.append(url)
        return body[url.rsplit("/", 1)[1]]

    t = datetime(2026, 10, 18, 20, tzinfo=timezone.utc)
    [(s, folder, created)] = P.fetch_pbp(tmp_path, [2026], now=lambda: t, get=get, get_json=lambda u: meta)
    assert created and len(calls) == 1
    m = json.loads((folder / "manifest.json").read_text())
    assert m["license"] == "CC-BY-4.0" and "look-ahead" in m["caveat"]
    [(_, folder2, created2)] = P.fetch_pbp(tmp_path, [2026], now=lambda: t, get=get, get_json=lambda u: meta)
    assert not created2 and folder2 == folder and len(calls) == 1  # no second download
    assert P.latest_files(tmp_path, [2026])[2026][1]["sha256"] == m["sha256"]
    (folder / "play_by_play_2026.parquet").write_bytes(b"y" * 10)
    with pytest.raises(ValueError, match="checksum"):
        P.latest_files(tmp_path, [2026])
