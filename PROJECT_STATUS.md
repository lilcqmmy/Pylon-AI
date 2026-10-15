# Project status

Living document. Original handoff status (2026-10-15) is preserved in `docs/handoff/PROJECT_STATUS.md`.

## 2026-10-15 — Stage 0/1 skeleton (synthetic only)

### Built and verified
- Python package `sfi` (Python 3.14 venv; deps: numpy, pyyaml, pytest). `pytest`: **27 passed**.
- Seeded synthetic league generator: 32 fictional teams, seasons 2021–2024, 17 weeks, 1,088 games.
  Hidden true win probabilities computed exactly and stored separately for evaluation only.
- Point-in-time data access (`known_results`), schema validation, CSV storage with SHA-256 snapshot IDs.
- Models: Elo (home field, margin of victory, season regression), home-win-rate and coin-flip baselines.
- Walk-forward backtest by week; quality gate with configurable sparse-data abstention (off by default).
- Immutable hashed forecast records in an append-only JSONL archive; re-running the backtest
  reproduced all 3,264 records byte-identically.
- Report: Brier, log loss, week-block bootstrap CIs, paired comparisons, calibration slope, reliability table.

### Baseline results — SYNTHETIC, not real NFL
Scored 2022–2024 (2021 burn-in), 784 non-tied games:

| Model | Brier | 95% CI | Log loss |
|---|---:|---|---:|
| baseline_elo | 0.2334 | [0.2238, 0.2435] | 0.6601 |
| home_rate | 0.2476 | [0.2440, 0.2512] | 0.6884 |
| coin_flip | 0.2500 | – | 0.6931 |
| oracle (true probs; ceiling) | 0.2201 | [0.2109, 0.2305] | 0.6303 |

Elo beats both naive baselines (paired CI excludes zero) and trails the oracle by ~0.013 Brier.
Elo calibration slope 0.76 → forecasts are too extreme with the untuned default parameters.
Parameters were **not** tuned; any tuning must use validation seasons only, with a held-out season untouched.

### Not built / not verified
- No real NFL data, data licenses, providers or credentials.
- No score/margin/total or player-stat distributions (Stage 2).
- No database, API, dashboard, agents/orchestrator, scheduling, model registry (Stages 3–4).
- Local git repository only (branch `main`, first commit `de593b9`); no remote, no CI.

### Open decisions (owner)
- Real data source and its license (candidate to evaluate: nflverse open data; verify terms before use).
- Run environment (local Windows PC assumed for now), budget.
- First real target after win probability: margins/totals or a specific player stat?
- Abstention thresholds: to be set from empirical studies, not guessed.

### Next steps
1. Optional: add a remote (e.g. private GitHub repo) and CI running pytest.
2. Tune Elo (K, home field, regression) on validation seasons; add a held-out season; add logistic recalibration.
3. Add score-margin and total distributions with interval-coverage tests.
4. Pick and license-check a real historical data source; write its adapter with `known_at` timestamps.
