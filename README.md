# Sports Forecasting Intelligence

Research-only (no-money) NFL forecasting with calibrated probabilities, strict point-in-time
discipline and walk-forward evaluation. Numeric forecasts come from statistical models, never from an LLM.

**Current state:** Stage 1 skeleton running on **synthetic** data only. See [PROJECT_STATUS.md](PROJECT_STATUS.md).
The original product brief is in [docs/handoff/](docs/handoff/).

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\sfi.exe synth      # fictional seasons -> data/synthetic/
.\.venv\Scripts\sfi.exe tune       # Elo grid search on validation seasons -> artifacts/elo_tuned.json
.\.venv\Scripts\sfi.exe backtest   # walk-forward forecasts -> archive/, report -> reports/
```

Configuration: `config/config.example.yaml` (copy to `config/config.yaml` for local overrides; that file is git-ignored).

## Layout

```
src/sfi/
  config.py                 YAML config; refuses llm_for_numeric_probability: true
  data/schema.py            Game record, validation, CSV I/O, known_results(as_of) point-in-time gate
  data/synthetic.py         Seeded fictional league + exact "truth" probabilities (evaluation only)
  models/base.py            WinModel interface (observe / predict / inputs)
  models/elo.py             Elo with home field, margin of victory, season regression
  models/baselines.py       Coin flip, historical home-win rate
  models/calibration.py     Platt recalibration refit walk-forward on past issued forecasts
  forecasts/record.py       Hashed immutable forecast records, append-only JSONL archive
  evaluation/walkforward.py Week-by-week backtest + quality gate (abstention)
  evaluation/tuning.py      Elo grid search; holdout seasons are dropped before it runs
  evaluation/metrics.py     Brier, log loss, reliability table, calibration slope, block bootstrap
  evaluation/report.py      Scoring + Markdown/JSON report
  cli.py                    `sfi synth`, `sfi tune`, `sfi backtest`
tests/                      Leakage, metrics, archive immutability, schema, models, end-to-end
```

## Rules the code enforces

- Models learn only via `observe()`, fed by `known_results(games, as_of)`; tests prove tampering with
  future results cannot change earlier forecasts.
- A forecast record cannot be created at or after kickoff.
- The archive never rewrites a forecast: identical re-runs are no-ops, changed content raises.
- Every record carries `data_kind` and `simulation_only`; synthetic reports are labeled as such.
- Ties are excluded from binary scoring and counted separately.
- Seasons are split burn-in < validation < holdout (checked at startup). Tuning sees only data up
  to the last validation season; `backtest` refuses a tuned artifact from a different data snapshot.
- Model versions encode their parameters, and forecast IDs encode the data snapshot.
