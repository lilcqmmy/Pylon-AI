# Sports Forecasting Intelligence

Research-only (no-money) NFL forecasting with calibrated probabilities, strict point-in-time
discipline and walk-forward evaluation. Numeric forecasts come from statistical models, never from an LLM.

**Current state:** Elo win-probability baseline running on real NFL schedules/results (nflverse,
2000–2025) and on a synthetic test league. The real-data holdout (2020–2025) is sealed. See
[PROJECT_STATUS.md](PROJECT_STATUS.md). The original product brief is in [docs/handoff/](docs/handoff/).

Data: nflverse (nflverse-data, CC-BY 4.0); schedules maintained by Lee Sharpe (nfldata).
Non-commercial research use; raw files are not redistributed (`data/raw/` is git-ignored).
Source evaluation: [docs/data_sources/nflverse_schedules.md](docs/data_sources/nflverse_schedules.md).

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q

# Real NFL data
.\.venv\Scripts\sfi.exe fetch                          # new nflverse snapshot (network) -> data/raw/
.\.venv\Scripts\sfi.exe tune --dataset nflverse        # Elo grid search on validation seasons
.\.venv\Scripts\sfi.exe backtest --dataset nflverse    # validation report; holdout stays sealed
.\.venv\Scripts\sfi.exe backtest-scores --dataset nflverse   # margin/total distributions -> reports/nflverse_scores_backtest.md

# Play-by-play team features (experimental; not used live)
.\.venv\Scripts\sfi.exe fetch --pbp 2000-2026          # ~480 MB first time; later only changed seasons
.\.venv\Scripts\sfi.exe backtest --dataset nflverse --features pbp   # reports/nflverse_pbp_backtest.md
.\.venv\Scripts\sfi.exe fetch --injuries 2009-2026 --depth-charts 2001-2026
.\.venv\Scripts\sfi.exe backtest --dataset nflverse --features qb    # reports/nflverse_qb_backtest.md

# Synthetic test league
.\.venv\Scripts\sfi.exe synth
.\.venv\Scripts\sfi.exe tune --dataset synthetic
.\.venv\Scripts\sfi.exe backtest --dataset synthetic --reveal-holdout
```

Prospective (live) forecasts for upcoming real games:

```powershell
.\.venv\Scripts\sfi.exe fetch                # results must be current, or forecast refuses to run
.\.venv\Scripts\sfi.exe forecast --dry-run   # preview
.\.venv\Scripts\sfi.exe forecast             # append to ledger/prospective_forecasts.jsonl
.\.venv\Scripts\sfi.exe score                # after a later fetch: score forecasts whose games are final
```

`forecast` uses the frozen tuned Elo (win probability) and the team points model (score-margin and
game-total distributions, full pmf stored), replays all history up to now, and archives each forecast with
its run (snapshot, code commit, tuned-parameter hash, market line at issue time as a benchmark). It
refuses to run if the snapshot is over 24h old or any game is missing a result that should be known.
`score` uses the latest forecast issued before each game's actual kickoff (Brier/log loss for win
probability; CRPS, MAE and interval coverage for margin and total, vs market lines at issue). Commit `ledger/` after each
run so git holds an outside record of when forecasts were made.

`scripts\live_run.ps1` runs one full cycle (fetch → forecast → score → commit `ledger/`) and logs to
`logs\live_runs.log`. Two Claude desktop scheduled tasks run it: `nfl-live-forecast-weekday`
(Mon and Thu 3:00 PM ET), `nfl-live-forecast-sunday` (Sun 8:30 AM ET) and
`nfl-live-forecast-sunday-evening` (Sun 6:30 PM ET). They only run while the app
is open; a missed run happens on the next launch.

`--reveal-holdout` evaluates the holdout seasons and appends an entry to `archive/holdout_reveals.jsonl`.
For real data, use it only once the model design is frozen.

Configuration: `config/config.example.yaml` (copy to `config/config.yaml` for local overrides; that file is git-ignored).

## Layout

```
src/sfi/
  config.py                 YAML config; refuses llm_for_numeric_probability: true
  datasets.py               Named datasets, season splits, benchmarks
  data/schema.py            Game record, validation, CSV I/O, known_results(as_of) point-in-time gate
  data/synthetic.py         Seeded fictional league + exact "truth" probabilities (evaluation only)
  data/sources/nflverse.py  Snapshot fetch + manifest, normalization, kickoff repair, timeline checks
  data/sources/nflverse_pbp.py  Per-season play-by-play snapshots; skips unchanged upstream files
  features/pbp.py           Team-game offense/defense aggregates; point-in-time rolling/season features
  models/pbp_logit.py       Elo + play-by-play features, ridge logistic refit each season (walk-forward)
  features/qb.py            QB game stats, starters, injury rows with explicit known-at times
  models/qb_adjust.py       Elo + QB availability (expected starter at as-of), ridge logistic walk-forward
  distributions.py          Discrete predictive distributions: CRPS, pinball, intervals, PIT, Wilson
  models/scores.py          Margin/total models: naive, Elo-margin, team points ratings (walk-forward spreads)
  evaluation/scores_eval.py Score walk-forward, coverage tests, report sections
  models/base.py            WinModel interface (observe / predict / inputs)
  models/elo.py             Elo: home field (off at neutral sites), margin of victory, season regression,
                            ratings keyed by franchise
  models/baselines.py       Coin flip, historical home-win rate
  models/calibration.py     Platt recalibration refit walk-forward on past issued forecasts
  forecasts/record.py       Hashed immutable forecast records, append-only JSONL archive
  evaluation/walkforward.py Per-kickoff-slate backtest + quality gate (abstention)
  evaluation/tuning.py      Elo grid search; holdout seasons are dropped before it runs
  evaluation/metrics.py     Brier, log loss, reliability table, calibration slope, block bootstrap
  evaluation/report.py      Scoring, benchmarks, paired comparisons, Markdown/JSON report
  live.py                   Prospective forecasting (freshness gates, run log) and ledger scoring
  cli.py                    `sfi synth | fetch | tune | backtest | forecast | score`
ledger/                     Prospective forecasts and run log (append-only, tracked in git)
tests/                      Leakage, metrics, archive immutability, schema, adapter, models, end-to-end
```

## Rules the code enforces

- Models learn only via `observe()`, fed only results whose known-at time is <= the forecast's as-of
  time; tests prove tampering with later results cannot change earlier forecasts.
- A forecast record cannot be created at or after kickoff.
- The archive never rewrites a forecast: identical re-runs are no-ops, changed content raises.
- Every record carries `data_kind` (`synthetic` or `real`); reports are labeled accordingly.
- Ties are excluded from binary scoring and counted separately.
- Seasons are split burn-in < validation < holdout (checked at startup). Tuning never loads holdout
  seasons; `backtest` doesn't load them unless `--reveal-holdout` is given, and refuses a stale tuned artifact.
- Market lines and weather never reach models; the market is scored only as a benchmark.
- Raw snapshots are immutable and checksummed; a tampered file fails to load.
- Model versions encode their parameters, and forecast IDs encode the data snapshot.
