from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sfi.data.sources import openmeteo as OM

from .conftest import make_game

ROOT = Path(__file__).resolve().parents[1]
KICK = datetime(2026, 10, 25, 17, 0, tzinfo=timezone.utc)
NOW = KICK - timedelta(hours=50)


def _fake(calls):
    def get(url):
        calls.append(url)
        times = [(KICK + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(-1, 4)]
        return {"hourly": {"time": times, "temperature_2m": [50, 48, 47, 46, 45], "wind_speed_10m": [10, 12, 14, 16, 30],
                           "wind_gusts_10m": [15, 20, 25, 22, 40], "precipitation": [0, 0.1, 0, 0, 0],
                           "precipitation_probability": [10, 40, 30, 20, 0]}}
    return get


def test_game_weather_uses_kickoff_window_and_imperial_units():
    calls = []
    w = OM.game_weather(41.86, -87.62, KICK, now=NOW, get_json=_fake(calls))
    assert w["temperature_2m_kickoff"] == 48 and w["wind_speed_10m_mean3h"] == pytest.approx(14.0)
    assert w["wind_gusts_10m_max3h"] == 25 and w["lead_hours"] == 50.0
    assert "temperature_unit=fahrenheit" in calls[0] and "wind_speed_unit=mph" in calls[0]
    assert w["requested_at_utc"] == NOW.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_record_weather_skips_domes_and_unknown_stadiums_and_never_raises():
    g1 = make_game(gid="G1", start=KICK, hs=None, as_=None)
    g2 = make_game(gid="G2", home="C", away="D", start=KICK, hs=None, as_=None)
    g3 = make_game(gid="G3", home="E", away="F", start=KICK, hs=None, as_=None)
    g4 = make_game(gid="G4", home="G", away="H", start=KICK, hs=None, as_=None)
    venues = {"G1": {"stadium_id": "CHI98", "roof": "outdoors"}, "G2": {"stadium_id": "DET00", "roof": "dome"},
              "G3": {"stadium_id": "XXX00", "roof": "outdoors"}, "G4": {"stadium_id": "BAL00", "roof": ""}}
    stadiums = OM.load_stadiums(ROOT / "data/reference/stadiums.csv")

    def flaky(url):
        if "39.2780" in url:
            raise TimeoutError("network down")
        return _fake([])(url)

    got, skipped = OM.record_weather([g1, g2, g3, g4], venues, stadiums, NOW, get_json=flaky)
    assert set(got) == {"G1"} and got["G1"]["stadium_id"] == "CHI98"
    assert skipped["G2"] == "dome" and "no coordinates" in skipped["G3"] and "TimeoutError" in skipped["G4"]


def test_stadium_coordinates_are_plausible():
    st = OM.load_stadiums(ROOT / "data/reference/stadiums.csv")
    for sid, s in st.items():
        if s["note"] != "international":
            assert 24 < s["lat"] < 49 and -125 < s["lon"] < -66, sid  # contiguous US
    assert len(st) >= 32
