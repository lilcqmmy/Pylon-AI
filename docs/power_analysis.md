# Minimum detectable effect (MDE) of the paired-Brier tests

Computed 2026-10-19 with `scripts/analysis/mde.py` from the validation backtest (2010–2019, nflverse
snapshot `8af1d73f…`). MDE = smallest true mean Brier improvement a two-sided test detects with **80% power**:
MDE = (z₁₋α/₂ + 0.84) × SE, where SE is the week-block bootstrap standard error of the mean per-game paired
difference (challenger − `elo_tuned`). The SE depends on how much a challenger's forecasts differ from
`elo_tuned`'s, so it is measured per comparison. Projections assume SE ∝ 1/√n.

| Comparison | Games | Mean diff | SE | MDE (95%) | MDE (98.3%, Bonferroni 3) | MDE at 500 games | MDE at 1,000 games | Games needed for 0.002 (95% / 98.3%) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `elo_avail`, all games | 2,662 | −0.0007 | 0.00064 | **0.0018** | 0.0021 | 0.0041 | 0.0029 | 2,100 / 2,800 |
| `elo_avail`, either team 2+ starters out | 1,007 | −0.0006 | 0.00146 | 0.0041 | 0.0047 | 0.0058 | 0.0041 | 4,200 / 5,600 |
| `elo_qbadj`, all games | 2,662 | −0.0004 | 0.00094 | **0.0026** | 0.0030 | 0.0060 | 0.0043 | 4,600 / 6,100 |
| `elo_qbadj`, a team's starting QB changed | 506 | −0.0008 | 0.00385 | 0.0108 | 0.0125 | 0.0108 | 0.0077 | 14,700 / 19,600 |
| `elo_hfa`, all games | 2,662 | +0.0002 | 0.00020 | 0.0006 | 0.0007 | 0.0013 | 0.0009 | 220 / 290 |

## Plain answers

- **Is a 0.002 gain (about a quarter of the market gap) detectable on 2010–2019?** For `elo_avail`, only just:
  the MDE is 0.0018 at 95%, and 0.0021 at the Bonferroni level, i.e. *not* reliably. For `elo_qbadj`,
  **no** (MDE 0.0026). On the pre-specified subsets, **no**: MDEs of 0.004–0.011.
- **So single-feature validation on 2010–2019 is barely informative** for effects of the size these features
  could plausibly have. A "no clear difference" result is expected even if the feature helps by 0.001–0.002.
  It does rule out large effects (≥ 0.002–0.003) and catches broken models.
- **`elo_hfa` is different**: its forecasts barely differ from `elo_tuned` (SE 0.0002), so small effects are
  detectable, but the effect it can produce is also small.
- **The pre-registered 500-game prospective decisions can only detect large effects**: at 500 games the MDE
  is about 0.004 (`elo_avail`) to 0.006 (`elo_qbadj`); detecting 0.002 would need roughly 2,100–6,100
  scored games (about 8–23 seasons). The most likely outcome at 500 games is "keep running". The registered
  rules stay as registered; this note sets expectations.
- Implication for future work: features need effects of ≥ 0.004 Brier to be confirmable within a couple of
  seasons, or several small effects must be combined into one pre-registered model (see `elo_combo`).
