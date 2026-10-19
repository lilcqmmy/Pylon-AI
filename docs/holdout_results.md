# Holdout results (2020–2025), revealed once on 2026-10-19

Design frozen in `docs/holdout_freeze.md` (commit `65e08df`, 2026-10-19 13:34:54 ET) before the reveal.
Elo was re-tuned on validation seasons only for the new schedule snapshot: parameters identical to the
frozen ones. Reveals logged in `archive/holdout_reveals.jsonl` at 17:35:15Z (win), 17:35:30Z (scores) and
17:36:43Z (players). Real NFL results, replayed under the known-at policy (not live forecasts).
Full tables: `reports/nflverse_backtest.md`, `reports/nflverse_scores_backtest.md`,
`reports/nflverse_players_backtest.md`.

## 1–2. Win probability (`elo_tuned`), 1,688 non-tied games

| | Holdout 2020–2025 | Validation 2001–2019 |
|---|---:|---:|
| Brier `elo_tuned` | **0.2230** [0.2169, 0.2293] | 0.2194 |
| Log loss | 0.6367 | 0.6283 |
| Calibration intercept / slope | −0.128 / 1.047 | +0.024 / 1.069 |
| Brier `home_rate` | 0.2494 | 0.2449 |
| Brier market (benchmark) | 0.2100 | 0.2117 |

- `elo_tuned` − `home_rate`: **−0.0265** [−0.0323, −0.0207] → clearly better.
- `elo_tuned` − market: **+0.0130** [+0.0093, +0.0167] → market clearly better; the gap is about twice
  the validation gap (+0.0070).
- **Main miscalibration: home-field advantage shrank.** Mean forecast for home teams (non-neutral) was
  ≈ 0.567 every season, while home teams won 0.504 (2020), 0.514 (2021), 0.575 (2022), 0.566 (2023),
  0.536 (2024), 0.536 (2025). The fixed 48-point Elo home edge, learned on 2001–2019, is too large for
  this era, which is the negative intercept and the over-forecast middle bins (0.5–0.6: 0.552 forecast
  vs 0.495 observed).

## 3–5. Score margin and total (`scores_points_empirical`), 1,693 games

| | Margin | Total |
|---|---:|---:|
| CRPS model / naive | **7.33** / 7.93 | **7.58** / 7.76 |
| Model − naive (CRPS) | −0.60 [−0.73, −0.46] | −0.18 [−0.28, −0.08] |
| MAE model / market line | 10.19 / 9.77 | 10.67 / 10.31 |
| Model − market (abs. error) | +0.43 [+0.29, +0.57] | +0.36 [+0.24, +0.48] |
| Coverage 50/80/90 (obs/claimed) | .521/.521, .806/.812, .901/.906 ✓ | .525/.522, .818/.810, .914/.910 ✓ |
| PIT χ² (9 df; 16.9 = 5% limit) | 8.6 ✓ | 5.0 ✓ |

- Implied P(home win) Brier from the margin distribution: 0.2234 (vs 0.2230 for `elo_tuned`), consistent.
- Intervals and distribution shape stay calibrated out of sample. The points model learns its home edge
  continuously, so it is less exposed to the home-field decline than Elo.

## 6. Player yards (`players_opp_eff`)

| Stat | Forecasts | CRPS opp × eff / recent avg | Difference | Coverage 50/80/90 (obs/claimed) | PIT χ² |
|---|---:|---:|---|---|---:|
| Passing | 3,333 | 53.06 / 52.77 | +0.28 [−0.15, +0.74] n.s. | .491/.506, .797/.805, .901/.904 ✓ | 16 ✓ |
| Rushing | 11,721 | **13.59** / 13.94 | −0.35 [−0.44, −0.25] better | .534/.525, .830/.817, .920/.913 ✗ (too wide) | 18 ✗ |
| Receiving | 26,512 | **13.37** / 13.72 | −0.35 [−0.40, −0.30] better | .531/.527, .822/.820, .915/.914 ✓ | 15 ✓ |

Pool players with no opportunity (scored 0): passing 315 (9.5%), rushing 2,519 (21%), receiving 5,650 (21%).

## Conclusions

- Every frozen model's validation conclusions held up out of sample: Elo beats the naive baseline;
  margin/total distributions beat naive and stay calibrated; opportunity × efficiency beats the recent
  average for rushing and receiving but not passing.
- The market remains clearly better at win probability (+0.013 Brier) and point spreads/totals (≈ 0.4
  points MAE).
- The clearest fixable weakness is Elo's fixed home-field advantage. Any fix (e.g. learning the home edge
  over time, as the points model does) must be judged on **new** data: prospective 2026+ forecasts, not
  this holdout, which has now been used.
