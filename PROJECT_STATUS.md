# Project status

Living document. Original handoff status (2026-10-15) is preserved in `docs/handoff/PROJECT_STATUS.md`.
Real-data results below are **retrospective backtests** (replayed under a known-at policy), not live
forecasts. Synthetic results come from a fictional league and only test the machinery.

## Current state (2026-10-15)

### Built and verified
- Python package `sfi` (Python 3.14 venv; deps: numpy, pyyaml, tzdata, pytest). `pytest`: **51 passed**.
- **Real data:** nflverse schedules adapter (owner confirmed non-commercial research use with CC-BY
  attribution on 2026-10-15). Immutable checksummed snapshots with manifests; `sfi fetch` for new pulls.
  Snapshot in use: retrieved 2026-10-15T19:19:32Z, SHA-256 `cf5b5f6b…`, 7,289 games 2000–2026.
- Normalization: Eastern kickoff → UTC (`tzdata`), franchise continuity (OAK→LV, SD→LAC, STL→LA),
  neutral sites (102), playoffs included, `result_known_at = kickoff + 5h`, consistency checks, and a
  per-team timeline check (each result known before that team's next kickoff).
- **Upstream defect found and repaired:** 101 night games from 2000–2005 are stored as `09:00` instead of
  `21:00` Eastern, which would have made results look known about 12 hours before kickoff. Fixed by a
  narrow, logged rule (102 rows including one neutral-site game). Not yet reported to nflverse.
- Market moneylines (de-vigged) scored as a **benchmark only**; market and weather columns never reach models.
- Walk-forward now forecasts each **kickoff slate** 1h before kickoff (a Sunday-night forecast can use
  Sunday-afternoon results once known).
- Elo: no home-field advantage at neutral sites; ratings keyed by franchise.
- **Sealed holdout:** `backtest` doesn't load holdout seasons unless `--reveal-holdout` is passed; each
  reveal is logged to `archive/holdout_reveals.jsonl`.
- Synthetic league (2017–2024) retained for pipeline tests; tuning, Platt recalibration and the archive
  work as before.

### Real NFL results — validation 2001–2019 (holdout 2020–2025 sealed)
Splits fixed by the owner on 2026-10-15 before any real-data results were viewed. 5,055 scored games
(10 ties excluded); Elo was tuned on these same seasons, so its scores here are slightly optimistic.

| Model | Games | Brier | 95% CI | Log loss | Calib. slope |
|---|---:|---:|---|---:|---:|
| market_moneyline (benchmark) | 3,593 | 0.2117 | [0.2059, 0.2175] | 0.6106 | 1.01 |
| elo_tuned | 5,055 | 0.2194 | [0.2155, 0.2232] | 0.6283 | 1.07 |
| elo_tuned_platt | 5,055 | 0.2196 | [0.2154, 0.2236] | 0.6286 | 1.02 |
| elo_default | 5,055 | 0.2198 | [0.2156, 0.2240] | 0.6290 | 0.94 |
| home_rate | 5,055 | 0.2449 | [0.2429, 0.2467] | 0.6829 | n/a |
| coin_flip | 5,055 | 0.2500 | – | 0.6931 | n/a |

- Tuned Elo: K 20, home field 48, season regression **0.5** (default 0.33); not at a grid edge.
- **Elo clearly beats the home-rate baseline:** −0.0255 Brier, CI [−0.0290, −0.0218].
- **Tuning barely matters** on real data: −0.0004 vs the default, CI [−0.0011, +0.0002].
- **Elo is already well calibrated** (slope ≈ 1); Platt recalibration adds nothing measurable.
- **The market is clearly better:** Elo − market = +0.0070 Brier, CI [+0.0038, +0.0101], on the 3,593
  games with moneylines (mostly 2006+). The market reflects injuries, QB changes and news that a
  results-only Elo can't see. Caveat: the line's timestamp is undocumented and probably near kickoff.

### Play-by-play team features (2026-10-18; experimental, not used live)
- Downloaded nflverse play-by-play 2000–2026 (27 parquet files, 459 MB, CC-BY 4.0; owner-approved),
  checksummed per season; `sfi fetch --pbp` re-downloads only seasons that changed upstream.
- Per team and game, offense and defense: EPA/play, success rate, pass EPA, rush EPA, explosive-play
  rate (pass ≥ 20 yd, rush ≥ 10 yd), early-down pass rate, turnover rate, pressure rate (proxy: sack or
  QB hit per dropback). Each as a 4-game rolling value (crossing seasons) and season-to-date (falling
  back to last season early on), play-weighted, built only from games already released by the replay.
- Models: ridge logistic regression on the Elo logit plus home-minus-away differences, refit at each
  season start on past forecasts only (L2 = 10, fixed). `pbp_logit_epa` uses EPA/play only (4 diffs);
  `pbp_logit_full` uses all 32. Holdout play-by-play (2020–2025) is not loaded by the backtest.
- Caveat: nflverse EPA comes from a model trained on many seasons, so historical EPA has a small look-ahead.

Validation 2001–2019 (5,055 games), paired Brier differences:

| Comparison | Diff | 95% CI | Reading |
|---|---:|---|---|
| pbp_logit_epa − elo_tuned | −0.0004 | [−0.0015, +0.0007] | no clear difference |
| pbp_logit_full − elo_tuned | +0.0017 | [−0.0006, +0.0039] | no clear difference |
| pbp_logit_full − pbp_logit_epa | +0.0020 | [+0.0002, +0.0039] | EPA-only better |
| pbp_logit_epa − market | +0.0063 | [+0.0034, +0.0094] | market better |

**Conclusion:** team efficiency stats add no measurable skill over Elo, and the full feature set
overfits (calibration slope 0.89). The market's edge most likely comes from information neither model
has (injuries, QB changes, news). The feature models stay out of the live runs.

### Player availability (evaluated and built 2026-10-18; not promoted)
- See `docs/data_sources/nflverse_availability.md` (reproduce with `scripts/analysis/`). Downloaded
  nflverse injuries 2009–2026 (2.0 MB) and depth charts 2001–2026 (16.1 MB), owner-approved, CC-BY 4.0.
- Injury `date_modified` exists 2010–2024 and is pregame (median ≈ 47–55 h before kickoff); missing in
  2009 and absent from 2025. Weekly depth charts (≤ 2024) lag but aren't hindsight-filled; 2025+ depth
  charts are daily timestamped snapshots (QB1 = actual starter 91%). Schedule QB ids are hindsight.
- Signal (validation ≤ 2019 only): starting QB changes in ≈ 11% of team-games and moves results ≈ 7.5
  points of win probability against Elo. Upper bound with *hindsight* starter + QB quality: −0.0015
  Brier vs Elo (≈ a quarter of the market gap); the pregame injury-report proxy alone: −0.0003 (n.s.).
- Built `elo_qb` (expected starter from point-in-time injury reports, QB rating from prior
  qb_epa/dropback, ridge logistic on Elo). Validation 2010–2019: **−0.0002 Brier vs Elo, 95% CI
  [−0.0012, +0.0007], no clear difference**; helps on Out/Doubtful games (−0.0041, n = 131) but those
  are rare. Not promoted to live. A scaling bug for rare flags was found and fixed during validation.

### Score-margin and game-total distributions (Stage 2, 2026-10-18)
- `sfi backtest-scores`: walk-forward predictive distributions over integer margins (home − away) and
  totals; archived as quantile summaries (`archive/score_forecasts.jsonl`). Holdout sealed.
- Models: `scores_naive` (past outcomes only), `scores_elo_*` (margin mean from tuned Elo points edge,
  OLS refit each season), `scores_points_*` (team offense/defense scoring ratings; untuned defaults).
  Spread from each model's own past residuals: `*_normal` (discretized Normal) or `*_empirical` (residual
  shape, keeps key-number mass).
- Evaluation: CRPS, MAE (median), RMSE (mean), pinball loss, 50/80/90% central-interval coverage tested
  against each forecast's own claimed mass (95% Wilson interval), randomized-PIT χ² uniformity test,
  market spread/total lines as point benchmarks.

Validation 2001–2019 (5,065 games):

| Target | Best model | CRPS | vs naive | MAE | Market line MAE | Coverage 50/80/90 (obs/claimed) |
|---|---|---:|---|---:|---:|---|
| Margin | scores_points_empirical | 7.63 | −0.61 [−0.68, −0.53] | 10.63 | 10.41 | .528/.523, .816/.812, .905/.908 ✓ |
| Total | scores_points_empirical | 7.70 | −0.22 [−0.28, −0.15] | 10.85 | 10.66 | .535/.526, .812/.812, .908/.909 ✓ |

- Intervals are calibrated for every model (all ✓). Normal spreads fail the PIT shape test (χ² ≈ 26–50:
  key numbers), empirical spreads pass (χ² 4.9–9.0); CRPS is the same either way.
- Points model vs Elo on margin: −0.025 CRPS [−0.053, +0.004], not clearly better.
- Implied P(home win) from the margin distribution: Brier 0.2192, consistent with `elo_tuned` (0.2194).
- The market is still more accurate: model MAE − market MAE = +0.22 [+0.14, +0.30] (margin) and
  +0.20 [+0.13, +0.26] (total).

### Synthetic results (pipeline check only)
Synthetic holdout 2023–2024 (525 games): tuned Elo 0.2454 Brier, Platt 0.2449, default 0.2524, oracle
0.2351. The holdout was viewed once before the K grid was widened (from the validation edge signal only).

### Prospective forecasting (built 2026-10-18)
- **First live run:** 2026-10-18T19:38:24Z, run `90f9f0675c61`, snapshot `sha256:2384f5f3bc7c2eb5`,
  code `f5962f9`; 54 forecasts for 18 games (week 6 late games through week 7); ledger committed
  in `a6456ff`, 26 min before the first kickoff. Week 7 forecasts predate week 6 results, so
  re-run before those games; scoring uses the latest pre-kickoff forecast.
- `sfi forecast`: frozen tuned Elo (K 20, HFA 48, regression 0.5, chosen on 2001–2019), replays all
  history to the current time, and forecasts unplayed games in the next 8 days (at least 10 min before
  kickoff). Models archived: `elo_tuned` (primary), `elo_tuned_platt`, `home_rate`.
- Ledger `ledger/prospective_forecasts.jsonl` plus run log `ledger/forecast_runs.jsonl` (snapshot,
  code commit and dirty flag, tuned-parameter hash, market line at issue time as a benchmark).
- Refuses stale data: snapshot older than 24h, or any game whose result should be known but isn't.
  Verified: refuses the 2026-10-15 snapshot (72h old, SEA@DEN and HOU@JAX results missing).
- `sfi score`: latest pre-kickoff forecast per model and game; excludes forecasts issued after the
  actual kickoff; compares with the market at issue time and the snapshot's (likely closing) line.
- A test confirms a live forecast equals the backtest forecast for the same game and as-of time.
- **Automated (2026-10-18):** `scripts/live_run.ps1` (fetch → forecast → score → commit `ledger/`),
  run by Claude desktop scheduled tasks `nfl-live-forecast-weekday` (Mon and Thu 3:00 PM ET) and
  `nfl-live-forecast-sunday` (Sun 8:30 AM ET) and `nfl-live-forecast-sunday-evening` (Sun 6:30 PM ET),
  saved in Auto mode. They only run while the app is open.
  Each run forecasts all games in the next 8 days, so every game gets several forecasts and scoring
  uses the latest pre-kickoff one. The Sunday-evening run refreshes Sunday-night games with that
  afternoon's results. Gap: Thanksgiving 12:30 games are last forecast on Monday.

### Not built / not verified
- No live forecast has been scored yet (all 18 games pending).
- No weather-forecast features; no player-stat distributions; margin/total forecasts are not yet live.
- No database, API, dashboard, agents/orchestrator, scheduling, model registry (Stages 3–4).
- Local git repository only (branch `main`); no remote, no CI.

### Open decisions (owner)
- When to reveal the 2020–2025 holdout (only after the next model changes are frozen).
- Report the 09:00 kickoff defect upstream to nflverse/nfldata?
- Run environment (local Windows PC assumed for now), budget.
- Abstention thresholds: to be set from empirical studies, not guessed.

### Next steps
1. Start live runs: `sfi fetch` then `sfi forecast` before each slate; `sfi score` after results.
2. Neither team efficiency nor QB availability closed the market gap (≈ 0.007). Stage 2 margin/total
   distributions are built and calibrated. Options: player stat distributions (passing/rushing/receiving),
   adding margin/total forecasts to the live ledger, or revealing the holdout for an honest estimate.
3. Score-margin and total distributions with interval-coverage tests.
4. Optional: remote repository and CI running pytest.

## History
- 2026-10-15 `de593b9`: Stage 0/1 skeleton; untuned Elo on synthetic seasons 2021–2024.
- 2026-10-15 `e797848`: Elo tuning on validation seasons, untouched holdout, walk-forward Platt recalibration.
- 2026-10-15 `a981217`: nflverse schedules evaluated (license, coverage, known-at); adapter planned.
- 2026-10-15 `b11a2c5`: nflverse adapter built; first real-data validation results; holdout 2020–2025 sealed.
- 2026-10-18: `sfi forecast` / `sfi score` for prospective forecasts with freshness gates and a ledger.
- 2026-10-18 `80a216a`: play-by-play team features evaluated on validation; no gain over Elo.
- 2026-10-18 `f90b007`: injuries / depth charts / QB starters evaluated for availability features.
- 2026-10-18 `44fb78c`: `elo_qb` built and validated on 2010–2019; no gain over Elo; not promoted.
- 2026-10-18: Stage 2 margin/total distributions with coverage and PIT tests; validated on 2001–2019.
