# Product requirements

## North star
A continuously evaluated, high-integrity probabilistic sports forecasting platform that improves on meaningful held-out performance. Favor selective publication and honest uncertainty over maximal number of forecasts.

## Initial sport
NFL first, with separate sport-specific feature engineering/models later for NBA, MLB and NHL.

## Forecast categories
- Game: home/away/tie where relevant, win probability, predicted scores, expected score differential, distributions for margin and total points.
- Player: pass yards/attempts/completions, receiving yards/receptions/targets, rush yards/carries, touchdowns, relevant position stats, uncertainty quantiles and event probabilities.
- Team: offensive/defensive points, possessions/drives, offensive efficiency, turnovers, pace, conversion rates, matchup adjustments.
- Threshold outcomes: for an explicitly supplied statistical threshold, calculate P(stat > threshold), P(stat < threshold) with exact handling of push/ties and source/version of threshold. No claim that one category is inherently safe.

## User experience
- Upcoming events list, filter by league/date/event/outcome category.
- Forecast card showing probability, model version, as-of timestamp, median/mean and interval, confidence diagnostics, explanation and data coverage.
- Abstained forecasts visible with reasons: sparse data, stale injury data, unstable lineup, severe model disagreement, failed quality checks, poor calibration.
- Historical audit page showing original forecast and later actuals without rewriting history.
- Reliability dashboard by sport, season, outcome type, bucket, team/player segment and time window.

## Success criteria (set numerical targets after reliable baseline data)
- Beats simple sensible benchmarks on properly held-out seasons, not only training data.
- Calibration curves and proper scoring rules reported, with uncertainty bands.
- Reproducible walk-forward evaluation and zero identified point-in-time leakage.
- All final forecasts reproducible from versioned data snapshot and code/model artifacts.
- No fabricated data or forced forecasts when quality gates fail.

## Constraints
- No-money sports research/analytics framing from earlier plan.
- Highly reliable probability estimates, but never promise guaranteed outcomes.
- Safety/security: limit personally identifying data; sanitize untrusted source content and tool outputs; protect API keys; separate agent suggestions from deployment authorization.
