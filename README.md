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

# Synthetic test league
.\.venv\Scripts\sfi.exe synth
.\.venv\Scripts\sfi.exe tune --dataset synthetic
.\.venv\Scripts\sfi.exe backtest --dataset synthetic --reveal-holdout
```

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
  cli.py                    `sfi synth | fetch | tune | backtest`
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
