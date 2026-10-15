# Build roadmap

## Stage 0 — initialize and audit
- Check for any actual existing sports repo, dataset, data license and API account (none verified in this handoff).
- Record deployment environment, budget and data rights.
- Initialize version control, reproducible Python environment, formatting, tests, CI and README.

## Stage 1 — trustworthy NFL baseline
- Acquire appropriately licensed historical NFL schedule/results and play-by-play with known availability times.
- Implement normalized entities, immutable snapshots, transformations and validation.
- Build simple Elo/logistic win probability model with strictly forward-in-time backtests.
- Save forecast records and postgame scores; evaluate Brier/log loss/calibration against baselines.
- Create minimal CLI and report.

## Stage 2 — outcome distributions
- Add score-margin and game-total distributions, observed outcomes and interval tests.
- Add player opportunity/efficiency models for passing, receiving, rushing stats and role uncertainty.
- Add simulations with cross-player/team correlation where feasible.
- Add configurable, empirically validated abstention gates.

## Stage 3 — multi-agent system
- Add Claude coordinator with tools for querying verified database, running forecast jobs and displaying provenance.
- Add specialized research, player, matchup, context, evaluation and explanation agents.
- Validate prompts/tools against fabricated statistics, prompt injection and inappropriate model overrides.

## Stage 4 — controlled autonomous improvement
- Automated scheduled data refreshes, result scoring, data-drift warnings, challenger training, model registry and reproducible promotion gates.
- Dashboard: outcomes, probabilities, margins, yards, intervals, calibration, prediction archive and abstentions.

## Stage 5 — scale
- MLB/NBA/NHL with separate data contracts and baseline models.
- Parallelized processing, efficient warehouse queries, selective GPU use, observability and cost caps.

## First concrete engineering sprint
1. Scaffold `src/{data,features,models,evaluation,api,agents}` and `tests/`.
2. Add synthetic non-real football data fixture and immutable forecast record serializer.
3. Add time-based leakage tests and Brier/log-loss evaluation tests.
4. Add a minimal baseline model and rolling evaluation harness.
5. Document dataset provider candidates and verify legal usage before importing real records.
6. Produce an initial baseline report clearly labeled synthetic until real data is integrated.
