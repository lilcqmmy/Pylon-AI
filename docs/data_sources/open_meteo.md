# Data source evaluation: Open-Meteo pregame weather forecasts

Evaluated 2026-10-19 for one purpose: a **point-in-time** weather input for game totals. Status: **recording
only.** No model uses weather until at least a season of live-recorded forecasts exists.

## License and limits
- Free API for non-commercial use (this project is no-money research), no API key.
- Data licence CC BY 4.0. Attribution: "Weather data by Open-Meteo.com (CC BY 4.0)".
- Limits 600/min, 5,000/hour, 10,000/day, per IP; a live run makes about 15–20 calls.

## Can it be backtested point-in-time?
- **Historical Forecast API** (2021/22+): stitches the first hours of successive model runs into one series.
  That is close to "the forecast at kickoff", including runs published after a pregame as-of time, so it is
  **not** point-in-time for our forecasts.
- **Previous Runs API**: forecasts at fixed leads of 1–7 days (`*_previous_dayN`). Point-in-time safe for
  as-of times at least that far before kickoff, but most models are archived only from **January 2024**
  (GFS temperature from 2021), i.e. about 2.5 seasons, with 24 h+ leads.
- Conclusion: record the forecast ourselves at issue time from now on (clean, at our actual lead time),
  and optionally use Previous Runs for 2024–2025 later.

## What each live run records (`ledger/forecast_runs.jsonl`, `weather_at_issue`)
For every forecast game not listed as a dome: hourly forecast at the stadium (best-match model) for the
kickoff hour and the 3 hours from kickoff: temperature (°F), wind speed and gusts (mph), precipitation
(inch) and precipitation probability; plus request time, lead hours, stadium id and the schedule's roof
value (retractable roofs are recorded; their open/closed decision is not known in advance). Failures are
recorded as skip reasons and never stop a run.

## Stadium coordinates
`data/reference/stadiums.csv`: hand-entered approximate coordinates for current and international venues.
Adequate for 9–13 km weather grids; a test checks that US venues are inside the contiguous US. Verify
before any finer use.
