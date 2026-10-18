# Project status

Living document. Original handoff status (2026-10-15) is preserved in `docs/handoff/PROJECT_STATUS.md`.
Real-data results below are **retrospective backtests** (replayed under a known-at policy), not live
forecasts. Synthetic results come from a fictional league and only test the machinery.

## Current state (2026-10-15)

### Built and verified
- Python package `sfi` (Python 3.14 venv; deps: numpy, pyyaml, tzdata, pytest). `pytest`: **51 passed**.
- **Real data:** nflverse schedules adapter (owner confirmed non-commercial research use with CC-BY
  attribution on 2026-10-15). Immutable checksummed snapshots with manifests; `sfi fetch` for new pulls.
  Snapshot in use: retrieved 2026-10-15T19:19:32Z, SHA-256 `cf5b5f6b…`, 7,289 games 2000–2026.
- Normalization: Eastern kickoff → UTC (`tzdata`), franchise continuity (OAK→LV, SD→LAC, STL→LA),
  neutral sites (102), playoffs included, `result_known_at = kickoff + 5h`, consistency checks, and a
  per-team timeline check (each result known before that team's next kickoff).
- **Upstream defect found and repaired:** 101 night games from 2000–2005 are stored as `09:00` instead of
  `21:00` Eastern, which would have made results look known about 12 hours before kickoff. Fixed by a
  narrow, logged rule (102 rows including one neutral-site game). Not yet reported to nflverse.
- Market moneylines (de-vigged) scored as a **benchmark only**; market and weather columns never reach models.
- Walk-forward now forecasts each **kickoff slate** 1h before kickoff (a Sunday-night forecast can use
  Sunday-afternoon results once known).
- Elo: no home-field advantage at neutral sites; ratings keyed by franchise.
- **Sealed holdout:** `backtest` doesn't load holdout seasons unless `--reveal-holdout` is passed; each
  reveal is logged to `archive/holdout_reveals.jsonl`.
- Synthetic league (2017–2024) retained for pipeline tests; tuning, Platt recalibration and the archive
  work as before.

### Real NFL results — validation 2001–2019 (holdout 2020–2025 sealed)
Splits fixed by the owner on 2026-10-15 before any real-data results were viewed. 5,055 scored games
(10 ties excluded); Elo was tuned on these same seasons, so its scores here are slightly optimistic.

| Model | Games | Brier | 95% CI | Log loss | Calib. slope |
|---|---:|---:|---|---:|---:|
| market_moneyline (benchmark) | 3,593 | 0.2117 | [0.2059, 0.2175] | 0.6106 | 1.01 |
| elo_tuned | 5,055 | 0.2194 | [0.2155, 0.2232] | 0.6283 | 1.07 |
| elo_tuned_platt | 5,055 | 0.2196 | [0.2154, 0.2236] | 0.6286 | 1.02 |
| elo_default | 5,055 | 0.2198 | [0.2156, 0.2240] | 0.6290 | 0.94 |
| home_rate | 5,055 | 0.2449 | [0.2429, 0.2467] | 0.6829 | n/a |
| coin_flip | 5,055 | 0.2500 | – | 0.6931 | n/a |

- Tuned Elo: K 20, home field 48, season regression **0.5** (default 0.33); not at a grid edge.
- **Elo clearly beats the home-rate baseline:** −0.0255 Brier, CI [−0.0290, −0.0218].
- **Tuning barely matters** on real data: −0.0004 vs the default, CI [−0.0011, +0.0002].
- **Elo is already well calibrated** (slope ≈ 1); Platt recalibration adds nothing measurable.
- **The market is clearly better:** Elo − market = +0.0070 Brier, CI [+0.0038, +0.0101], on the 3,593
  games with moneylines (mostly 2006+). The market reflects injuries, QB changes and news that a
  results-only Elo can't see. Caveat: the line's timestamp is undocumented and probably near kickoff.

### Synthetic results (pipeline check only)
Synthetic holdout 2023–2024 (525 games): tuned Elo 0.2454 Brier, Platt 0.2449, default 0.2524, oracle
0.2351. The holdout was viewed once before the K grid was widened (from the validation edge signal only).

### Prospective forecasting (built 2026-10-18; no live forecasts archived yet)
- `sfi forecast`: frozen tuned Elo (K 20, HFA 48, regression 0.5, chosen on 2001–2019), replays all
  history to the current time, and forecasts unplayed games in the next 8 days (at least 10 min before
  kickoff). Models archived: `elo_tuned` (primary), `elo_tuned_platt`, `home_rate`.
- Ledger `ledger/prospective_forecasts.jsonl` plus run log `ledger/forecast_runs.jsonl` (snapshot,
  code commit and dirty flag, tuned-parameter hash, market line at issue time as a benchmark).
- Refuses stale data: snapshot older than 24h, or any game whose result should be known but isn't.
  Verified: refuses the 2026-10-15 snapshot (72h old, SEA@DEN and HOU@JAX results missing).
- `sfi score`: latest pre-kickoff forecast per model and game; excludes forecasts issued after the
  actual kickoff; compares with the market at issue time and the snapshot's (likely closing) line.
- A test confirms a live forecast equals the backtest forecast for the same game and as-of time.

### Not built / not verified
- No live forecasts archived yet; all real-data scores are retrospective.
- No injury, QB, roster or weather-forecast features; no score/margin/total or player-stat distributions.
- No database, API, dashboard, agents/orchestrator, scheduling, model registry (Stages 3–4).
- Local git repository only (branch `main`); no remote, no CI.

### Open decisions (owner)
- Cadence for live runs (e.g. `fetch` + `forecast` Thursday afternoon and Sunday morning) and whether to automate it.
- When to reveal the 2020–2025 holdout (only after the next model changes are frozen).
- Report the 09:00 kickoff defect upstream to nflverse/nfldata?
- Run environment (local Windows PC assumed for now), budget.
- Abstention thresholds: to be set from empirical studies, not guessed.

### Next steps
1. Start live runs: `sfi fetch` then `sfi forecast` before each slate; `sfi score` after results.
2. Close the gap to the market with point-in-time features: QB starter changes and injury reports
   (nflverse injuries have `date_modified`; needs its own known-at evaluation).
3. Score-margin and total distributions with interval-coverage tests.
4. Optional: remote repository and CI running pytest.

## History
- 2026-10-15 `de593b9`: Stage 0/1 skeleton; untuned Elo on synthetic seasons 2021–2024.
- 2026-10-15 `e797848`: Elo tuning on validation seasons, untouched holdout, walk-forward Platt recalibration.
- 2026-10-15 `a981217`: nflverse schedules evaluated (license, coverage, known-at); adapter planned.
- 2026-10-15 `b11a2c5`: nflverse adapter built; first real-data validation results; holdout 2020–2025 sealed.
- 2026-10-18: `sfi forecast` / `sfi score` for prospective forecasts with freshness gates and a ledger.
