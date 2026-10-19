# Challenger pre-registration: `elo_hfa` (time-varying home-field advantage)

Registered 2026-10-19, before any prospective forecast from this model exists.

## Why
The 2020–2025 holdout (`docs/holdout_results.md`) showed home teams winning less often than `elo_tuned`
assumes (forecast ≈ 0.567; observed 0.50–0.58). Because the holdout motivated this change, it **cannot**
be used to judge it. The only fair test is games played after the challenger starts forecasting.

## Models
- Champion: `elo_tuned` (frozen: version `0.3.0+851aa1d0`, K 20, home edge fixed at 48 Elo points,
  season regression 0.5, margin of victory on).
- Challenger: `elo_hfa` (`sfi.models.elo.EloAdaptiveHome`): identical ratings and parameters, but the
  home edge starts at 48 and after each non-neutral game moves by `home_k × (outcome − expected)`,
  clipped to [0, 150]. **`home_k = 1.0`, fixed by reasoning, not fitted**: if home teams win about 4 points
  of probability less than expected, the edge drops by about 10 Elo points per season; with ≈ 256 home
  games a season, random noise moves it by about ±8 points.
- Both are replayed over all history to the forecast time with the same data and point-in-time rules,
  and archived side by side in the live ledger.

## Comparison (pre-specified)
- Data: prospective ledger forecasts (`ledger/prospective_forecasts.jsonl`) for games kicking off after
  the first run that includes `elo_hfa`; per model and game, the latest forecast issued before the actual
  kickoff (`sfi score`'s rule). Ties excluded.
- Primary metric: paired Brier difference `elo_hfa − elo_tuned`, 95% week-block bootstrap interval.
  Secondary: log loss, calibration intercept, mean forecast vs observed home-win rate.
- Decision, at the first `sfi score` with **≥ 500 scored games** (about two seasons):
  - upper CI bound < 0 → promote `elo_hfa` to primary (`live.primary_model`);
  - lower CI bound > 0 → retire `elo_hfa`;
  - otherwise keep both and re-check every additional 250 games.
- No change to either model, and no tuning of `home_k`, until the decision. The validation backtest is
  only a sanity check that the challenger is not broken; it is not evidence for it.

## Status log
- 2026-10-19: wired into `sfi forecast` / `sfi score` (live ledger, paired with `elo_tuned`). Validation
  sanity check only (not evidence): `elo_hfa` − `elo_tuned` = +0.0002 [−0.0001, +0.0005] on 2001–2019.
  Learned home edge over time (Elo points): 60.9 (Aug 2005), 51.3 (2011), 59.7 (2014), 47.4 (2020),
  35.1 (2023), 31.1 (2026-10-18) versus the frozen 48. First prospective forecasts: next scheduled run.
