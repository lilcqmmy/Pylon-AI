# Market-residual family: validation results (2006–2017)

Computed 2026-10-19 by `sfi market-validate` after the pre-registration (`docs/market_biases.md`, commit `6b487e1`)
and Amendment 1 (`ab26c3b`, e2 dropped as empty; only subset sizes had been seen). Full tables:
`reports/market_validation.md`. **Lines treated as the close (timing undocumented).** Holdout 2018–2025 sealed.
Research only; separate from the independent forecast.

## Pre-registered hypotheses (Holm family of 10)

| ID | Hypothesis | Games | Rate / difference [95% CI] | MDE (80% power) | p | Holm p | Paper return per unit [95% CI] |
|---|---|---:|---|---:|---:|---:|---|
| a1 | Home underdogs, home ATS | 1,019 | 49.9% [46.7, 53.0] | ±4.4 pts | 0.950 | 1.000 | −0.024 [−0.080, +0.031] |
| a2 | Home underdogs in primetime | 181 | 50.8% [43.3, 58.3] | ±10.4 pts | 0.882 | 1.000 | −0.010 [−0.141, +0.117] |
| j | Late-season home underdogs | 257 | 48.2% [42.0, 54.5] | ±8.7 pts | 0.618 | 1.000 | −0.053 [−0.150, +0.048] |
| b | Road favorites 7+: underdog ATS | 203 | 56.2% [49.0, 63.1] | ±9.8 pts | 0.092 | 0.827 | +0.076 [−0.051, +0.196] |
| c2 | Over rate, dome − outdoor | 466 vs 2,311 | +5.0 pts [+0.0, +9.9] (54.9% vs 50.0%) | ±7.1 pts | 0.051 | 0.509 | – |
| i | High totals: under | 399 | 50.6% [45.6, 55.6] | ±7.0 pts | 0.841 | 1.000 | −0.007 [−0.099, +0.089] |
| d | Underdog cover, divisional − other | 1,133 vs 1,982 | +0.7 pts [−3.0, +4.3] | ±5.2 pts | 0.717 | 1.000 | – |
| e1 | Team off a bye vs rested opponent | 372 | 52.2% [46.9, 57.3] | ±7.3 pts | 0.437 | 1.000 | +0.018 [−0.079, +0.118] |
| f | Underdog ATS at spreads of exactly 3 or 7 | 671 | 52.2% [48.3, 56.0] | ±5.4 pts | 0.280 | 1.000 | +0.019 [−0.049, +0.087] |
| k | Fade a team that covered its last 3 | 527 | 50.5% [46.1, 54.8] | ±6.1 pts | 0.862 | 1.000 | −0.017 [−0.100, +0.065] |

**No hypothesis is supported** (all Holm p ≥ 0.5), so none goes to the holdout under the registered rule.
Read with the MDEs: most subsets can only detect deviations of 5–10 points from 50%, while beating the vig needs
2.4 points; so "not supported" means "no large bias detectable", not "no bias". The two closest (b, c2) are
consistent with the literature's direction but nowhere near the corrected threshold.

## market_logit

| Target | Market-alone calibration (intercept, slope) | Walk-forward 2008–2017: market_logit − market (Brier) [95% CI] | MDE |
|---|---|---|---:|
| Home win (de-vigged moneyline) | −0.041 [−0.112, +0.031], **1.015 [0.909, 1.105]** | +0.0013 [−0.0000, +0.0026] | 0.0019 |
| Home ATS cover (de-vigged spread prices) | −0.046 [−0.107, +0.024], slope not identifiable* | +0.0008 [−0.0007, +0.0022] | 0.0020 |
| Over (de-vigged total prices) | +0.022 [−0.044, +0.089], slope not identifiable* | +0.0011 [−0.0002, +0.0024] | 0.0018 |

\*Spread and total prices barely vary around 50%, so a slope on them cannot be estimated (CIs span about −1 to +1).

- The moneyline market is well calibrated (slope ≈ 1).
- Adding the indicators makes forecasts **slightly worse** out of sample on every target: the indicators add
  noise, not signal.
- Exploratory only (not a registered test, one of 23 coefficients): the `road_fav7` coefficient for ATS cover is
  +0.268 [+0.035, +0.510], echoing hypothesis b. With 23 coefficients, one interval excluding zero is expected by
  chance.

## Decision
Nothing passes validation, so nothing goes to the holdout (which stays sealed). `market_logit` is wired live
(item 15) **for tracking only**, labelled as a market-residual forecast. Prospective data (time-stamped at-issue vs
close lines, live weather) is the remaining test, including the live-only hypotheses c1 and g.
