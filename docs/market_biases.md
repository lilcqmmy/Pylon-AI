# Pre-registration: market-bias hypotheses (market-residual family)

Registered 2026-10-19, **before any against-the-spread (ATS), over/under or market-bias statistic has been computed in
this project** (see the contamination record in `docs/data_sources/market_lines.md`). Research and paper tracking only.

## Ground rules
- Family: market-residual models may use market lines as inputs; the independent forecast (`elo_tuned` and challengers)
  never does. The families are reported separately and never blended.
- Lines: nflverse has one line per game, updated in place, with undocumented timing; historical lines are
  **treated as the close**. Live runs record lines at issue time from 2026.
- Splits: **validation 2006–2017**; **holdout 2018–2025, sealed** until the owner says otherwise (reveals logged, as
  before); live (prospective) from 2026. Exception for (h): see below.
- Neutral-site games are excluded from every home/road hypothesis. Playoffs are included unless stated.

## Outcomes and conventions
- `s` = `spread_line` (home favored by `s`), `m` = home margin. **Home covers** if `m > s`, **push** if `m = s`, away
  covers if `m < s`. Underdog = side with `s` against it; pick'em (`s = 0`) excluded from favorite/underdog hypotheses.
- **Over** if total points > `total_line`, push if equal.
- Pushes are excluded from rates and count as 0 return.
- Prices: actual `*_spread_odds` / `over_odds` / `under_odds` / moneylines; −110 when missing (2006, 2008, 2009 partly).
  Break-even at −110 is 110/210 = **52.38%**.
- Primetime: kickoff 19:00 ET or later. Rest days: nflverse `home_rest` / `away_rest`. Bye: rest ≥ 13. Short rest: ≤ 4.

## Tests (pre-specified)
- **Primary:** rate of the hypothesized side (cover / under / over) vs 50%: exact two-sided binomial test,
  Clopper–Pearson 95% interval. Difference hypotheses: two-proportion z-test.
- **Secondary:** paper return per unit at the prices above, 1 unit per game on the hypothesized side, 95% week-block
  bootstrap interval. A profitable bias must clear 52.38% (at −110), not just 50%.
- **Every result reports its minimum detectable effect** (80% power at its level): MDE ≈ (z + 0.84) × √(0.25 / n).
- **Multiplicity:** Holm correction across all hypotheses testable on validation (list below). A hypothesis is
  "supported in validation" only if its Holm-adjusted p < 0.05 in the stated direction (two-sided hypotheses: either).
- **Holdout:** only supported hypotheses go to the holdout, tested one-sided in the validated direction at
  α = 0.05 / (number carried). Holdout stays sealed until the owner unseals it.

## Hypotheses

| ID | Subset (exact) | Bet side / outcome | Direction expected | Data | Basis |
|---|---|---|---|---|---|
| a1 | Home underdogs: `s < 0`, non-neutral | home ATS cover | > 50% | 2006–2017 / holdout | Golec & Tamarkin 1991; Levitt 2004 |
| a2 | a1 and primetime | home ATS cover | > 50% | 2006–2017 / holdout | as a1 (high-attention games) |
| j | a1 and in the last 4 weeks of that regular season (week ≥ season's last REG week − 3) | home ATS cover | > 50% | 2006–2017 / holdout | Borghesi 2007 (late-season home-underdog bias) |
| b | Road favorites of 7+: `s ≤ −7`, non-neutral | home (underdog) ATS cover | > 50% | 2006–2017 / holdout | Levitt 2004 (favorites over-bet) |
| c1 | Wind ≥ 15 mph in the **pregame forecast recorded at issue** (non-dome) | under | > 50% | live 2026+ only | weather and scoring; tested at the first season end with ≥ 100 games |
| c2 | Dome (`roof = dome`) vs outdoor (`roof = outdoors`); retractable/blank excluded | over rate, dome minus outdoor | ≠ 0 (two-sided) | 2006–2017 / holdout | Borghesi 2007 (weather/venue effects) |
| i | High totals: `total_line` ≥ 90th percentile of totals in the previous 3 seasons | under | > 50% | 2006–2017 / holdout | Paul & Weinbach 2002 (unders on the highest totals) |
| d | Divisional games (`div_game = 1`) vs non-divisional | underdog ATS cover rate, divisional minus non-divisional | ≠ 0 (two-sided); also divisional underdog vs 50% | 2006–2017 / holdout | familiarity hypothesis (no specific citation) |
| e1 | Team off a bye (rest ≥ 13) vs opponent not off a bye | that team's ATS cover | ≠ 50% (two-sided) | 2006–2017 / holdout | rest effects (no specific citation) |
| e2 | Team on short rest (≤ 4) vs opponent not on short rest | that team's ATS cover | ≠ 50% (two-sided) | 2006–2017 / holdout | rest effects (no specific citation) |
| f | Key numbers: `|s|` exactly 3 or 7 vs all other non-zero spreads | underdog ATS cover (pushes excluded) | > 50% at key numbers; also key minus non-key ≠ 0 | 2006–2017 / holdout | favorite over-pricing (Levitt 2004) at key margins |
| g | Large line move: live spread at close differs from the first recorded at-issue spread by ≥ 1.5 points | cover of the side the line moved toward | ≠ 50% (two-sided) | live 2026+ only | Moskowitz 2021 (overreaction in line moves); tested at the first season end with ≥ 100 games |
| h | Elo–market disagreement 0.05 ≤ \|`elo_tuned` − market P(home)\| < 0.10 (market = de-vigged moneyline) | Elo's side: moneyline paper return at the close; ATS cover of Elo's side | > 0 return; > 50% cover | **2020–2025 only** (see note) | item 10 hypothesis; Moskowitz 2021 |
| k | Team covered ATS in each of its previous 3 games (same season, computed from past results only) | that team's ATS cover | < 50% (fade) | 2006–2017 / holdout | Moskowitz 2021 (momentum overreaction reversed by outcomes) |

Validation family for Holm (testable retrospectively): a1, a2, j, b, c2, i, d, e1, e2, f, k (11 tests).
Live-only: c1, g. Separate: h.

**Note on (h):** item 10 already reported Elo-vs-market paper returns by disagreement bucket for 2006–2019, so
2006–2019 are not independent for (h). (h) is therefore **not** tested on validation and not part of the Holm family;
its only retrospective test is on **2020–2025** (Elo-vs-market Brier was revealed there, but disagreement returns
were not), one-sided at 0.05, when the owner unseals the holdout. It is also tracked live.

## What is not a hypothesis
`market_logit` (item 14) is a modelling exercise over the same features; its coefficients are reported with
intervals but are not additional tests. Exploratory looks at other subsets must be labelled exploratory and cannot
be promoted without a new pre-registration and fresh data.

## References
- Golec, J. & Tamarkin, M. (1991). The degree of inefficiency in the football betting market: Statistical tests.
  *Journal of Financial Economics* 30(2), 311–323.
- Gray, P. K. & Gray, S. F. (1997). Testing market efficiency: Evidence from the NFL sports betting market. *Journal of
  Finance* 52(4), 1725–1737. (Also: documented inefficiencies tend to fade over time.)
- Paul, R. J. & Weinbach, A. P. (2002). Market efficiency and a profitable betting rule: Evidence from totals on
  professional football. *Journal of Sports Economics* 3(3), 256–263.
- Levitt, S. D. (2004). Why are gambling markets organised so differently from financial markets? *Economic Journal*
  114(495), 223–246.
- Borghesi, R. (2007). The late-season bias: explaining the NFL's home-underdog effect. *Applied Economics* 39(15),
  1889–1903.
- Borghesi, R. (2007). The home team weather advantage and biases in the NFL betting market. *Journal of Economics and
  Business* 59(4), 340–354.
- Moskowitz, T. J. (2021). Asset pricing and sports betting. *Journal of Finance* 76, 3153–3209.

Published effects largely predate 2006, and Gray & Gray note they tend to fade, so the prior expectation is that
most of these will **not** survive in 2006–2017 data.

## Amendment 1 (2026-10-19, before any outcome was computed)
While building the subsets, only subset **sizes** were inspected (no covers, totals or returns). Hypothesis **e2**
(team on short rest vs an opponent not on short rest) selects **0 games** in 2006–2017: on Thursday games both teams
are on short rest, so the registered subset is empty by construction. e2 is **dropped as untestable**, with no
replacement (choosing a new definition now would be post hoc; a rest hypothesis can be re-registered for live data).
The validation Holm family is therefore 10 tests: a1, a2, j, b, c2, i, d, e1, f, k. Subset sizes seen: a1 1,050;
a2 189; j 267; b 205; c2 476 dome vs 2,355 outdoor; i 407; d 1,167 divisional vs 2,032 other; e1 384; f 725 key vs
2,474 other; k 539.
