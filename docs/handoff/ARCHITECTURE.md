# Proposed architecture — design, not existing implementation

```
Licensed/statistical feeds + historical play-by-play + official context
        ↓
Source adapters → event-normalization → freshness/quality checks
        ↓
Raw immutable snapshots / versioned feature tables (as-of timestamps)
        ↓
 ┌─────────────────────────────────────────────────────┐
 │ Claude orchestration / research and explanation      │
 │ collection | matchup | player usage | injuries        │
 │ simulation | validation | audit/monitoring agents      │
 └─────────────────────────────────────────────────────┘
        ↓ deterministic feature definitions
Baseline models + sport-specific ML + calibrated ensembles
        ↓ predictive distributions + uncertainty
Quality gate / abstention → API → dashboard / history
        ↓ (after event finishes)
Outcome ingestion → scoring → drift checks → candidate retraining
        ↓ gated promotion if tested improvement
Model registry (immutable) / champion-challenger
```

## Agents and responsibilities
- **Coordinator (Claude):** delegates bounded analyses, assembles cited narratives, compares conflicting findings, never invents stats or probabilities.
- **Data integrity agent:** source freshness, duplicates, season alignment, missing records, suspicious values and provenance.
- **Historical/statistical agent:** rolling features, recency adjustments, opponent strength, strength of schedule and multi-year priors.
- **Matchup/context agent:** team schemes, weather, home/away, travel, rest, injuries and reported lineups.
- **Player usage agent:** snaps, routes, targets, carries, opportunities, role changes and depth-chart effects.
- **Modeling/simulation engine (deterministic Python):** generate probabilities and score/player-stat distributions. Simulations preserve correlations where possible.
- **Evaluation agent:** evaluates proper scores, errors, coverage, calibration, subgroup quality and drift; suggests changes, does not silently promote models.
- **Explanation agent:** provides source-backed reasons and caveats in user-friendly format.

## Suggested stack
- Python: numpy/pandas or Polars; scikit-learn; LightGBM/XGBoost; PyTorch only if justified; scipy/statsmodels for distributions and calibration.
- Data: PostgreSQL (+ optional TimescaleDB), Parquet, DuckDB; object storage for immutable raw snapshots.
- Scheduling/orchestration: Prefect or Dagster; start with simple local scripts for MVP.
- API: FastAPI; interface: Next.js; experiments/model registry: MLflow.
- Compute: CPUs sufficient for many baseline/tabular tasks; GPUs selectively for large neural models or extensive simulation.

## Critical separations
- LLM reasoning and narratives are not a substitute for calibrated numerical forecasts.
- Historical forecasting must replay pregame knowledge, not current retrospectively corrected data.
- Workflow automation is separate from authorizing deployments/model promotions.
- Different leagues need distinct rules, features, distributions and evaluations.
