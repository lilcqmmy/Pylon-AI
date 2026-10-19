"""Pregame weather forecasts from Open-Meteo, recorded at issue time (data collection only; no model uses it).

Open-Meteo free API: non-commercial use, data under CC BY 4.0, no API key, 10,000 calls/day.
Attribution: "Weather data by Open-Meteo.com (CC BY 4.0)".

Why record live: the Historical Forecast API stitches the first hours of successive model runs (not what
was available before kickoff), and the Previous Runs API (fixed 1-7 day leads, point-in-time safe) starts
in January 2024 for most models. Recording the forecast at issue time builds a clean point-in-time series.

Stadium coordinates: `data/reference/stadiums.csv`, hand-entered approximate coordinates (fine for 9-13 km
weather grids; verify before using for anything finer).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"
HOURLY = ["temperature_2m", "wind_speed_10m", "wind_gusts_10m", "precipitation", "precipitation_probability"]


def load_stadiums(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["stadium_id"]: {**r, "lat": float(r["lat"]), "lon": float(r["lon"])} for r in csv.DictReader(fh)}


def _get_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "sfi-research"})
    with urlopen(req, timeout=30) as resp:  # noqa: S310 (fixed https host)
        return json.loads(resp.read())


def game_weather(lat: float, lon: float, kickoff: datetime, *, now: datetime,
                 get_json: Callable[[str], dict] = _get_json) -> dict:
    """Hourly forecast at the kickoff hour and the mean/max over the 3 hours from kickoff (°F, mph, inch)."""
    start = kickoff.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=3)
    params = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "hourly": ",".join(HOURLY),
              "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch",
              "timezone": "UTC", "start_hour": start.strftime("%Y-%m-%dT%H:%M"),
              "end_hour": end.strftime("%Y-%m-%dT%H:%M")}
    data = get_json(FORECAST_URL + "?" + urlencode(params))
    h = data.get("hourly", {})
    times = h.get("time", [])
    window = [i for i, t in enumerate(times) if start <= datetime.strptime(t, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc) < end]
    if not window:
        raise ValueError("forecast has no hours for the kickoff window")

    def vals(k):
        return [h[k][i] for i in window if h.get(k) and h[k][i] is not None]

    out = {"requested_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "lat": round(lat, 4), "lon": round(lon, 4),
           "model": "best_match", "lead_hours": round((start - now).total_seconds() / 3600, 1)}
    for k in HOURLY:
        v = vals(k)
        out[f"{k}_kickoff"] = v[0] if v else None
        out[f"{k}_mean3h"] = round(sum(v) / len(v), 3) if v else None
    gusts = vals("wind_gusts_10m")
    out["wind_gusts_10m_max3h"] = max(gusts) if gusts else None
    return out


def record_weather(games, venues: dict[str, dict], stadiums: dict[str, dict], now: datetime,
                   get_json: Callable[[str], dict] = _get_json) -> tuple[dict[str, dict], dict[str, str]]:
    """Weather for non-dome games. Returns ({game_id: weather}, {game_id: skip reason}). Never raises."""
    got, skipped = {}, {}
    for g in games:
        v = venues.get(g.game_id, {})
        if v.get("roof") == "dome":
            skipped[g.game_id] = "dome"
            continue
        st = stadiums.get(v.get("stadium_id", ""))
        if st is None:
            skipped[g.game_id] = f"no coordinates for stadium {v.get('stadium_id')!r}"
            continue
        try:
            got[g.game_id] = {**game_weather(st["lat"], st["lon"], g.scheduled_start_utc, now=now, get_json=get_json),
                              "stadium_id": v.get("stadium_id"), "roof_in_schedule": v.get("roof") or None}
        except Exception as e:  # noqa: BLE001 (weather is optional; record why it is missing)
            skipped[g.game_id] = f"{type(e).__name__}: {e}"[:200]
    return got, skipped
