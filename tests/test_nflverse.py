"""nflverse adapter. Fixture rows are hand-written in nflverse's column format (not copied data)."""

import json
from datetime import datetime, timezone

import pytest

from sfi.data.schema import ValidationError
from sfi.data.sources import nflverse as N
from sfi.models.elo import EloModel

COLS = ("game_id,season,game_type,week,gameday,weekday,gametime,away_team,away_score,home_team,home_score,"
        "location,result,total,away_moneyline,home_moneyline,spread_line,temp,wind").split(",")


def row(gid, season, week, gameday, gametime, away, home, a_sc="", h_sc="", location="Home",
        game_type="REG", away_ml="", home_ml="", weekday="Sunday"):
    r = dict.fromkeys(COLS, "NA")
    final = a_sc != "" and h_sc != ""
    r.update(game_id=gid, season=str(season), game_type=game_type, week=str(week), gameday=gameday,
             weekday=weekday, gametime=gametime, away_team=away, home_team=home, location=location,
             away_score=a_sc or "NA", home_score=h_sc or "NA",
             result=str(int(h_sc) - int(a_sc)) if final else "NA",
             total=str(int(h_sc) + int(a_sc)) if final else "NA",
             away_moneyline=away_ml or "NA", home_moneyline=home_ml or "NA")
    return r


def norm(rows, min_season=2000):
    return N.normalize_rows(rows, min_season=min_season, known_at_hours=5)


def test_eastern_time_to_utc_across_dst():
    # 2024 DST ended 2024-11-03: EDT (UTC-4) before, EST (UTC-5) after.
    assert N.kickoff_utc("2024-10-27", "13:00") == datetime(2024, 10, 27, 17, tzinfo=timezone.utc)
    assert N.kickoff_utc("2024-11-03", "13:00") == datetime(2024, 11, 3, 18, tzinfo=timezone.utc)
    # 2025 DST started 2025-03-09 (no games), check a January (EST) and September (EDT) game.
    assert N.kickoff_utc("2025-01-05", "20:20") == datetime(2025, 1, 6, 1, 20, tzinfo=timezone.utc)
    assert N.kickoff_utc("2025-09-07", "09:30") == datetime(2025, 9, 7, 13, 30, tzinfo=timezone.utc)


def test_normalizes_scores_franchises_neutral_and_known_at():
    games, market, repairs = norm([
        row("2019_01_SD_OAK", 2019, 1, "2019-09-09", "22:15", "SD", "OAK", "20", "24", weekday="Monday"),
        row("2024_06_JAX_CHI", 2024, 6, "2024-10-13", "09:30", "JAX", "CHI", "16", "35", location="Neutral"),
        row("2026_10_KC_DEN", 2026, 10, "2026-11-15", "16:25", "KC", "DEN"),
    ])
    oak, lon, fut = games
    assert (oak.home_team, oak.home_key, oak.away_key) == ("OAK", "LV", "LAC")
    assert oak.result_known_at_utc == oak.scheduled_start_utc.replace(hour=7)  # 02:15Z kickoff + 5h
    assert oak.known_at_policy == "kickoff_plus_5h" and oak.home_outcome == 1.0
    assert lon.neutral_site and lon.scheduled_start_utc.hour == 13
    assert fut.home_score is None and fut.result_known_at_utc is None and fut.known_at_policy is None
    assert repairs == [] and market == {}


def test_market_columns_are_not_on_games_and_are_devigged():
    games, market, _ = norm([row("2015_01_A", 2015, 1, "2015-09-13", "13:00", "NYJ", "NE", "10", "20",
                                 away_ml="+150", home_ml="-170")])
    g = games[0]
    for leaky in ("home_moneyline", "spread_line", "temp", "wind", "total"):
        assert not hasattr(g, leaky)
    ph, pa = 170 / 270, 100 / 250
    assert market["2015_01_A"] == pytest.approx(ph / (ph + pa))
    assert N.market_probability("NA", "-110") is None


def test_pre_2006_nine_am_night_games_are_repaired_but_london_games_are_not():
    games, _, repairs = norm([
        row("2003_01_NYJ_WAS", 2003, 1, "2003-09-04", "09:00", "NYJ", "WAS", "13", "16", weekday="Thursday"),
        row("2015_04_NYJ_MIA", 2015, 4, "2015-10-04", "09:30", "NYJ", "MIA", "27", "14", location="Neutral"),
    ])
    assert games[0].scheduled_start_utc == datetime(2003, 9, 5, 1, tzinfo=timezone.utc)  # 21:00 EDT
    assert games[1].scheduled_start_utc == datetime(2015, 10, 4, 13, 30, tzinfo=timezone.utc)
    assert len(repairs) == 1 and repairs[0].startswith("2003_01_NYJ_WAS")


def test_rejects_inconsistent_or_unknown_rows():
    bad = row("2010_01_X", 2010, 1, "2010-09-12", "13:00", "NYJ", "BUF", "10", "20")
    bad["result"] = "3"
    with pytest.raises(ValidationError, match="result"):
        norm([bad])
    with pytest.raises(ValidationError, match="unknown team"):
        norm([row("2010_01_Y", 2010, 1, "2010-09-12", "13:00", "XYZ", "BUF", "10", "20")])
    with pytest.raises(ValidationError, match="missing gametime"):
        norm([row("2010_01_Z", 2010, 1, "2010-09-12", "NA", "NYJ", "BUF", "10", "20")])


def test_timeline_check_catches_result_known_after_next_kickoff():
    with pytest.raises(ValidationError, match="kick off <3 days apart"):
        norm([row("2010_01_A", 2010, 1, "2010-09-12", "13:00", "NYJ", "BUF", "10", "20"),
              row("2010_02_A", 2010, 2, "2010-09-13", "20:30", "NYJ", "MIA", "10", "20", weekday="Monday")])


def test_min_season_filters_rows():
    games, _, _ = norm([row("1999_01_A", 1999, 1, "1999-09-12", "NA", "MIN", "ATL", "17", "14")])
    assert games == []


def test_elo_ignores_home_field_at_neutral_site_and_follows_franchise():
    games, _, _ = norm([
        row("2019_01_SD_OAK", 2019, 1, "2019-09-09", "22:15", "SD", "OAK", "20", "24", weekday="Monday"),
        row("2020_01_LV_LAC", 2020, 1, "2020-09-13", "16:05", "LV", "LAC", location="Neutral"),
    ])
    m = EloModel()
    m.start_season(2019)
    m.observe(games[0])
    m.start_season(2020)
    # Ratings carried over from OAK/SD to LV/LAC; neutral site removes home edge.
    assert m.ratings["LV"] > 1500 > m.ratings["LAC"]
    assert m.predict_home_win(games[1]) < 0.5


def _fake_get(payloads):
    def get(url):
        return payloads[url.rsplit("/", 1)[1]]
    return get


def test_fetch_creates_immutable_snapshot_and_skips_identical(tmp_path):
    payload = {"games.csv": b"game_id,season\nx,2026\n", "timestamp.json": b'{"last_updated":"2026-10-15 15:16:13 EDT"}'}
    t1 = datetime(2026, 10, 15, 19, 0, tzinfo=timezone.utc)
    folder, m, created = N.fetch_schedules(tmp_path, now=lambda: t1, get=_fake_get(payload))
    assert created and folder.name == "20261015T190000Z"
    assert m["license"] == "CC-BY-4.0" and m["upstream_last_updated"] == "2026-10-15 15:16:13 EDT"
    assert json.loads((folder / "manifest.json").read_text())["sha256"] == m["sha256"]
    # Same bytes later: no new folder.
    t2 = datetime(2026, 10, 16, 19, 0, tzinfo=timezone.utc)
    folder2, _, created2 = N.fetch_schedules(tmp_path, now=lambda: t2, get=_fake_get(payload))
    assert not created2 and folder2 == folder
    # Changed bytes: new folder, latest points at it.
    payload["games.csv"] += b"y,2026\n"
    folder3, _, created3 = N.fetch_schedules(tmp_path, now=lambda: t2, get=_fake_get(payload))
    assert created3 and N.latest_snapshot(tmp_path) == folder3
    with pytest.raises(FileExistsError):
        N.register_snapshot(folder, retrieved_at=t1, source_url="x")


def test_load_snapshot_rejects_tampered_file(tmp_path):
    payload = {"games.csv": b"game_id,season\n", "timestamp.json": b"{}"}
    folder, _, _ = N.fetch_schedules(tmp_path, get=_fake_get(payload))
    (folder / "games.csv").write_bytes(b"tampered\n")
    with pytest.raises(ValidationError, match="checksum"):
        N.load_snapshot(folder, min_season=2000, known_at_hours=5)
