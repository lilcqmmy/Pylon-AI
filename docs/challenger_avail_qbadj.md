# Challenger pre-registration: `elo_avail` and `elo_qbadj`

Registered 2026-10-19, before either model has produced a prospective forecast. Both were developed after
the 2020–2025 holdout was spent; validation 2010–2019 results are sanity checks only:
`elo_avail` − `elo_tuned` −0.0007 [−0.0019, +0.0005]; `elo_qbadj` − `elo_tuned` −0.0004 [−0.0022, +0.0014].

## Frozen definitions (no changes until a decision)

**`elo_avail`** (`sfi.models.avail.EloAvail`): ridge logistic (L2 10, refit each season start on past
forecasts) on the tuned-Elo logit and the home-minus-away weighted count of expected starters listed
Out/Doubtful on an injury report known by the forecast time. Expected starters: depth-team-1 players
(same-week weekly chart through 2024; latest daily snapshot ≤ as-of from 2025). Groups and weights, fixed by
reasoning: QB 3.0; OL (T, G, C, LT, LG, RT, RG, OT, OG, OL) 1.0; secondary (CB, S, SS, FS, DB, NB, LCB, RCB)
1.0; pass rush (DE, DT, NT, OLB, LDE, RDE, LDT, RDT, EDGE, DL, LOLB, ROLB) 0.7; skill (WR, TE, RB, HB) 0.5.

**`elo_qbadj`** (`sfi.models.qbadj.EloQbAdj`): tuned Elo with each team's rating adjusted by
875 × (expected starter's rating − team's season QB baseline), used for the forecast and the rating update.
QB rating = prior `qb_epa` per dropback shrunk to 0 with 300 dropbacks; expected starter = first
depth-chart QB not listed Out/Doubtful before as-of (previous starter if no chart); baseline =
dropback-weighted mean rating of this season's starters (last season early on).

## Comparison (pre-specified, same data rules as `docs/challenger_elo_hfa.md`)

- Data: prospective ledger forecasts for games kicking off after the first run that includes the
  challenger; per model and game the latest forecast issued before the actual kickoff; ties excluded.
- Primary metric: paired Brier difference challenger − `elo_tuned` over all scored games, week-block
  bootstrap.
- Pre-specified subsets (reported, not used for the decision): `elo_avail` — either team has 2+ expected
  starters Out/Doubtful at forecast time; `elo_qbadj` — either team's expected starter at forecast time
  differs from its previous game's starter.
- Multiplicity: three challengers (`elo_hfa`, `elo_avail`, `elo_qbadj`) are compared with the same
  champion. For `elo_avail` and `elo_qbadj` the decision uses a **98.3% interval** (95% Bonferroni-adjusted
  for 3 comparisons). `elo_hfa` keeps its earlier registered 95% rule.
- Decision at the first `sfi score` with **≥ 500 scored games** for that challenger:
  - upper bound < 0 → eligible for promotion to `live.primary_model`;
  - lower bound > 0 → retire;
  - otherwise keep running and re-check every additional 250 games.
- If more than one challenger becomes eligible at the same check, promote the one with the lowest mean
  Brier difference; combining challengers requires a new pre-registration.
- Promotion changes only which model is primary; the frozen `elo_tuned` keeps running as a comparison.
