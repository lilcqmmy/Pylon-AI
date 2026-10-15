# Project status

Living document. Original handoff status (2026-10-15) is preserved in `docs/handoff/PROJECT_STATUS.md`.
Everything below is **synthetic** (fictional teams and games); no real NFL data has been used.

## Current state (2026-10-15)

### Built and verified
- Python package `sfi` (Python 3.14 venv; deps: numpy, pyyaml, pytest). `pytest`: **36 passed**.
- Seeded synthetic league: 32 fictional teams, seasons 2017–2024, 17 weeks, 2,176 games. Hidden true
  win probabilities are computed exactly and stored separately for evaluation only.
- Point-in-time data access, schema validation, CSV snapshots with SHA-256 IDs.
- Models: Elo (home field, margin of victory, season regression), home-win-rate and coin-flip baselines,
  and a Platt recalibration wrapper refit before each week on past issued forecasts only.
- Season splits enforced in config: burn-in 2017, validation 2018–2022, holdout 2023–2024.
- `sfi tune`: Elo grid search (360 configurations) scored only on validation seasons; holdout seasons
  are dropped before tuning runs, and a test proves holdout results cannot change the chosen parameters.
- Walk-forward backtest by week; optional sparse-data abstention gate (off by default).
- Append-only hashed forecast archive; model versions encode parameters, forecast IDs encode the data
  snapshot. Re-running the backtest reproduced all records byte-identically.
- Report: validation and holdout sections with Brier, log loss, week-block bootstrap CIs, paired
  comparisons, calibration slope and reliability tables.

### Tuned Elo (selected on validation seasons)
K = 8, home field = 48 Elo, season regression = 0.33, margin-of-victory on (default had K = 20).
K = 8 is inside the grid (2–30), not at its edge.

### Results — SYNTHETIC, not real NFL

Holdout 2023–2024 (525 non-tied games; never used for tuning):

| Model | Brier | 95% CI | Log loss | Calib. slope |
|---|---:|---|---:|---:|
| elo_tuned_platt | 0.2448 | [0.2363, 0.2531] | 0.6823 | 0.78 |
| elo_tuned | 0.2454 | [0.2368, 0.2535] | 0.6837 | 0.87 |
| home_rate | 0.2499 | [0.2460, 0.2537] | 0.6930 | n/a |
| coin_flip | 0.2500 | – | 0.6931 | n/a |
| elo_default | 0.2524 | [0.2405, 0.2643] | 0.6991 | 0.45 |
| oracle (true probs; ceiling) | 0.2351 | [0.2229, 0.2471] | 0.6622 | 0.82 |

Paired holdout comparisons (week-block bootstrap):
- Tuned vs default Elo: −0.0070 Brier, CI [−0.0119, −0.0021] → **tuning clearly helps**.
- Platt vs tuned Elo: −0.0006, CI [−0.0024, +0.0012] → **no clear difference**; the tuned Elo is already
  close to calibrated, so recalibration adds little here.
- Tuned Elo vs home-rate: −0.0045, CI [−0.0119, +0.0030] → **not distinguishable on the holdout**
  (it is clearly better on validation: −0.0122, CI [−0.0167, −0.0072]).
- The holdout seasons are harder than validation (oracle Brier 0.2351 vs 0.2236), and 525 games is a
  small sample: even the true probabilities show a calibration slope of 0.82 there, so slopes from
  this sample are noisy.

Process note: the holdout numbers were viewed once before the K grid was extended from 8–30 to 2–30.
That change was triggered by the best K sitting at the grid edge on validation; it moved holdout Brier
by 0.0003.

### Not built / not verified
- No real NFL data, data licenses, providers or credentials.
- No score/margin/total or player-stat distributions (Stage 2).
- No database, API, dashboard, agents/orchestrator, scheduling, model registry (Stages 3–4).
- Local git repository only (branch `main`); no remote, no CI.

### Open decisions (owner)
- Real data source and its license (candidate to evaluate: nflverse open data; verify terms before use).
- Run environment (local Windows PC assumed for now), budget.
- First real target after win probability: margins/totals or a specific player stat?
- Abstention thresholds: to be set from empirical studies, not guessed.

### Next steps
1. Add score-margin and total distributions with interval-coverage tests.
2. Pick and license-check a real historical data source; write its adapter with `known_at` timestamps.
   Real data matters more than further synthetic tuning: these synthetic results only show the
   machinery works.
3. Optional: add a remote (e.g. private GitHub repo) and CI running pytest.

## History
- 2026-10-15 `de593b9`: Stage 0/1 skeleton; untuned Elo on seasons 2021–2024.
- 2026-10-15: Elo tuning on validation seasons, untouched holdout, walk-forward Platt recalibration.
