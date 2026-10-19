from datetime import timedelta

import pytest

from sfi import picks as PK

from .conftest import T0, make_game


def _game(gid="2026_07_NE_CHI", home="CHI", away="NE", hs=None, as_=None, start=T0):
    return make_game(gid=gid, home=home, away=away, season=2026, week=7, hs=hs, as_=as_, start=start)


def _run(lines, issued=T0 - timedelta(hours=5), gids=("2026_07_NE_CHI",)):
    return {"game_ids": list(gids), "market_lines_at_issue": lines, "run_id": "r1",
            "issued_at_utc": issued.strftime("%Y-%m-%dT%H:%M:%SZ")}


def test_american_odds_conversions():
    assert PK.american_implied(-150) == pytest.approx(0.6)
    assert PK.american_implied(+150) == pytest.approx(0.4)
    assert PK.american_profit(+150, 1) == pytest.approx(1.5)
    assert PK.american_profit(-200, 1) == pytest.approx(0.5)


def test_pick_side_edge_stake_and_confidence():
    g = _game()
    lines = {g.game_id: {"home_ml": -150, "away_ml": +130}}  # raw implied 0.600 / 0.435
    [row] = PK.make_picks(run=_run(lines), p_home={g.game_id: 0.48}, games={g.game_id: g})
    # home edge 0.48 - 0.60 = -0.12; away edge 0.52 - 0.4348 = +0.085
    assert row["Pick"] == "NE" and row["Pick Side"] == "away" and row["Odds (American)"] == "+130"
    assert float(row["Edge"]) == pytest.approx(0.52 - 100 / 230, abs=1e-4)
    assert row["Stake (units)"] == 1 and row["Confidence"] == "Medium"
    assert row["Pick ID"] == "2026-07-NE-CHI-ML" and row["Paper Only"] == "yes"
    assert "not wagering advice" in row["AI Reasoning (summary)"]
    [small] = PK.make_picks(run=_run({g.game_id: {"home_ml": -120, "away_ml": +100}}), p_home={g.game_id: 0.55},
                            games={g.game_id: g})
    assert small["Stake (units)"] == 0 and small["Confidence"] == "Low"  # edges below 0.03


def test_missing_odds_still_writes_row_without_stake():
    g = _game()
    [row] = PK.make_picks(run=_run({}), p_home={g.game_id: 0.62}, games={g.game_id: g})
    assert row["Pick"] == "CHI" and row["Odds (American)"] == "" and row["Edge"] == ""
    assert row["Stake (units)"] == 0 and row["Confidence"] == "Low"


def test_write_appends_with_single_header(tmp_path):
    g = _game()
    rows = PK.make_picks(run=_run({}), p_home={g.game_id: 0.62}, games={g.game_id: g})
    p = PK.write_picks(rows, tmp_path, "2026-10-19")
    PK.write_picks(rows, tmp_path, "2026-10-19")
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("Pick ID,Date,Week,Sport,Away,Home,Bet Type,Pick,Line,Odds (American)")
    assert len(lines) == 3 and len(PK.read_all_picks(tmp_path)) == 2


def test_grading_uses_latest_pre_kickoff_pick_and_odds_at_issue(tmp_path):
    g_open = _game()
    early = PK.make_picks(run=_run({g_open.game_id: {"home_ml": -150, "away_ml": +130}}, issued=T0 - timedelta(days=2)),
                          p_home={g_open.game_id: 0.48}, games={g_open.game_id: g_open})
    late = PK.make_picks(run={**_run({g_open.game_id: {"home_ml": -110, "away_ml": -110}}, issued=T0 - timedelta(hours=2)),
                              "run_id": "r2"}, p_home={g_open.game_id: 0.60}, games={g_open.game_id: g_open})
    after_kick = PK.make_picks(run={**_run({g_open.game_id: {"home_ml": -110, "away_ml": -110}}, issued=T0 + timedelta(minutes=5)),
                                    "run_id": "r3"}, p_home={g_open.game_id: 0.10}, games={g_open.game_id: g_open})
    rows = early + late + after_kick
    final = _game(hs=24, as_=17)  # home win
    gr = PK.grade(rows, {final.game_id: final}, {final.game_id: {"home_ml": -130, "away_ml": +110}}, n_boot=100)
    # latest pre-kickoff pick: home at -110, edge 0.60 - 0.524 = 0.076 -> staked, won: +0.909 units
    assert gr["staked_bets"] == 1 and gr["units_profit"] == pytest.approx(100 / 110, abs=1e-3)
    # CLV: implied at close 130/230 = 0.565 minus at issue 0.524 > 0 (moved toward the pick)
    assert gr["clv_mean"] == pytest.approx(130 / 230 - 110 / 210, abs=1e-4)
    assert gr["picks_graded"] == 1 and gr["pick_hit_rate"] == 1.0


def test_tie_is_a_push_and_pending_games_are_counted():
    g = _game(hs=20, as_=20)
    rows = PK.make_picks(run=_run({g.game_id: {"home_ml": -110, "away_ml": -110}}), p_home={g.game_id: 0.65},
                         games={g.game_id: g})
    gr = PK.grade(rows, {g.game_id: g}, {}, n_boot=50)
    assert gr["units_profit"] == 0.0 and gr["staked_bets"] == 1
    g2 = _game()
    gr2 = PK.grade(rows, {g2.game_id: g2}, {}, n_boot=50)
    assert gr2["pending"] == 1 and gr2["staked_bets"] == 0
