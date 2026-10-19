# Holdout freeze (pre-registration)

Written and committed **before** the 2020–2025 holdout is revealed for the first time, on the owner's
instruction of 2026-10-19. The commit containing this file timestamps the frozen design. Nothing below
may be changed in response to holdout results; any later model change must be evaluated on new data
(prospective forecasts), not on this holdout.

## Frozen code, configuration and data

| Item | Value |
|---|---|
| Code commit (models and evaluation) | `372d61902c4e99f121bb784d7df9036cc0497641` |
| `config/config.example.yaml` SHA-256 | `e321ee436cac055b7a075e303b3068382e2c2888d52c73ecd16939e3d673aa89` |
| Schedule snapshot | `20261018T223846Z`, SHA-256 `8af1d73f78668999a1fc88216b358f121797fbbe15a485736fcb0f4303514c98` |
| Play-by-play 2000–2025 | combined id `sha256:1758eedf63398102` |
| Injuries 2009–2025 | combined id `sha256:7de6d9693a8d00ad` |
| Season splits | burn-in 2000; validation 2001–2019; **holdout 2020–2025** |

## Frozen models

| Target | Model | Version | Settings |
|---|---|---|---|
| Win probability | `elo_tuned` | `0.3.0+851aa1d0` | K 20, home field 48 Elo (0 at neutral sites), season regression 0.5, margin of victory on (chosen on 2001–2019) |
| Score margin and game total | `scores_points_empirical` | `0.1.0+85477dec` | k 0.06, k_league 0.005, k_home 0.003, season regression 0.4, init league 20.5, init home 2.5; empirical residual spread, window 2,000, min 200 |
| Player passing/rushing/receiving yards | `players_opp_eff` | `0.2.0+d453b4a8` | pool: last 2 games, min 10 att / 3 car / 2 tgt, single expected QB, not Out/Doubtful; k_player 150/80/40; k_def 300/150/300; volume decay 0.35; residual window 4,000, min_k 300 |

Before the reveal, Elo is re-tuned on validation seasons only, because the schedule snapshot changed
since tuning. **If the re-tuned parameters differ from those above, the reveal stops** and this is reported.

## Pre-specified reporting (all reported regardless of outcome)

Win probability, holdout 2020–2025 (`sfi backtest --dataset nflverse --reveal-holdout`):
1. Brier, log loss, calibration slope/intercept, reliability table for `elo_tuned`.
2. Paired Brier differences: `elo_tuned` − `home_rate`, `elo_tuned` − `market_moneyline` (benchmark).

Score margin and total (`sfi backtest-scores --dataset nflverse --reveal-holdout`):
3. CRPS, MAE, 50/80/90% coverage vs claimed mass (Wilson), PIT χ² for `scores_points_empirical`.
4. Paired: vs `scores_naive` (CRPS) and vs market spread/total lines (absolute error, benchmark).
5. Implied P(home win) Brier from the margin distribution.

Player yards (`sfi backtest-players --dataset nflverse --reveal-holdout`):
6. Per stat: CRPS, MAE, coverage vs claimed, PIT χ² for `players_opp_eff`; paired vs `players_recent_avg`.

Comparisons with validation are descriptive only. Context to keep in mind: 2020 had few or no fans
(home-field advantage expected to shrink), and 2021+ has 17-game seasons.

The holdout is revealed once. Every reveal is appended to `archive/holdout_reveals.jsonl`.
