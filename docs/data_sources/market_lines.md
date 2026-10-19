# Market-line data audit (nflverse schedules), 2006–2026

Audited 2026-10-19. Ground rule (owner decision 2026-10-19): market lines may be **inputs only to the
separate market-residual family** (`market_logit` and successors). The independent forecast (`elo_tuned`
and its challengers) never sees them. The two families are evaluated and reported separately, never blended.

## What exists
Only the `schedules` release (`games.csv`) carries betting data; none of the other 24 nflverse-data releases
(checked via the GitHub API: pbp, injuries, depth_charts, rosters, …) has lines. Columns, one value per game:

| Column | Meaning (nflreadr dictionary) |
|---|---|
| `spread_line` | Spread; positive = home favored by that many points; lines up with `result` |
| `total_line` | Total points line |
| `home_moneyline`, `away_moneyline` | American odds to win |
| `home_spread_odds`, `away_spread_odds` | American odds to cover (actual prices, e.g. −110, −105, −108, −102, −115) |
| `over_odds`, `under_odds` | American odds for over / under `total_line` |

There is **no opening line and no timestamp** anywhere in nflverse. The dictionary names no sportsbook or source
and does not say opening or closing.

## Recording time
- Lines are **updated in place**: between our snapshots of 2026-10-15 and 2026-10-18, 18 of 184 upcoming games
  changed (e.g. 2026_06_DAL_GB spread 1.5 → −1.5, home moneyline −120 → −108). Lines appear only about a week
  before kickoff (166–169 of the later 2026 games had none).
- So a historical value is most plausibly the **last update before kickoff, i.e. approximately the closing
  line**. Every market-family report labels historical lines "treated as close (timing undocumented)".
- From 2026 on, each live run's `market_lines_at_issue` (run log) and `ledger/market_moves.jsonl` record
  time-stamped lines, so a true issue-vs-close dataset accumulates. Week-over-week moves (hypothesis g) can only
  be tested on that live data.

## Missingness (final games)
| Seasons | spread_line, total_line | moneylines, spread odds, over/under odds |
|---|---|---|
| 2006 | complete | 18% missing |
| 2007 | complete | complete |
| 2008 | complete | 27% missing |
| 2009 | complete | 5% missing |
| 2010–2026 | complete | complete |

Spreads include half points in 44.8% of games. Where spread or total odds are missing, analyses that need a
price use −110 and say so.

## Prior use of these lines in this project (contamination record)
- Moneyline-implied probability: benchmark in validation 2001–2019 and in the 2020–2025 holdout reveal
  (Brier only); item-10 study (2006–2019) of Elo−market disagreement buckets including paper returns.
- Spreads/totals: used as point benchmarks for margin/total MAE (2001–2019, 2020–2025).
- **No against-the-spread cover, over/under outcome, or market-bias statistic has been computed** before the
  bias pre-registration (`docs/market_biases.md`).
