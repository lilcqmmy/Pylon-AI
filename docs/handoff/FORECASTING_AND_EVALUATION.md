# Forecasting, uncertainty, self-improvement and evaluation

## Baselines before complexity
1. Event win model: simple strength ratings (Elo-like), home field, injuries/rest as available; compare with logistic regression and boosted trees.
2. Points: estimate home and away scoring distributions and score-margin distribution; backtest mean/median and intervals.
3. Player volume models: passing attempts, targets, carries, snap share; combine opportunity and efficiency into player stat distributions.
4. Threshold probabilities are derived from these calibrated predictive distributions; handle discrete mass and ties.
5. Ensembles only after independently validating diversity and incremental gain.

## Leakage control
- Use rolling-origin / walk-forward splits grouped by event date and season.
- Feature source availability timestamp must be <= prediction_as_of. Late corrections cannot rewrite earlier information state.
- Preseason and early-season forecasts require shrinkage/hierarchical priors and visibly wider ranges.
- Refuse labels/target-derived predictors that incorporate the game being predicted.
- Hyperparameter selection on historical validation periods; final holdout remains untouched.

## Evaluation metrics
- Binary/categorical events: Brier score, log loss, reliability diagram, calibration slope/intercept, sharpness, sample counts.
- Point forecasts: MAE and RMSE for scores/margins/yards; relative to naive benchmarks.
- Distributions: proper scoring rules (CRPS where appropriate), quantile/pinball loss, interval coverage and width.
- Subgroups: position, season phase, weather, underdog/favorite equivalent by modeled probabilities, missing-data regimes.
- Abstention: report published coverage and accuracy/calibration *on published subset*, plus withheld subset. Evaluate selective risk and selection bias.
- Model uncertainty: bootstrap/time-blocked intervals, drift and small-sample alerts; do not label a few successes proof of calibration.

## Policy for high-reliability forecasts
- Probability threshold is configurable and **never guaranteed**. Empirically validate probabilities, particularly extreme tails.
- Reject if critical feed is stale, lineup unknown, missing feature percentage high, models substantially disagree, or historical coverage is too low.
- Distinguish “80% estimated chance of outcome” from “80% confidence in the estimate”.
- Keep point predictions separate from event probabilities; show distributions and intervals.

## Learning loop
1. Append immutable signed/hashed forecast record before event; snapshot inputs and version.
2. Ingest official final result separately, resolve stat corrections under versioned policies.
3. Score original forecast; generate residual/error diagnostics by group.
4. Diagnose drift, context misses and calibration error; avoid post-hoc storytelling.
5. On schedule, train challenger on incrementally larger past-only data.
6. Compare champion and challenger on rolling forward backtests and untouched holdout, with statistical uncertainty, latency and operational budgets.
7. Explicitly approve promotion; track rollback point and monitor after release.

## Correlation warning
Receiving yards, pass yards, point totals and game outcomes are correlated. Do not treat forecasts as independent. Joint simulations or empirically validated dependence structures are required for joint event probabilities.
