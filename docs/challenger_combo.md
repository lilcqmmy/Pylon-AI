# Challenger pre-registration: `elo_combo`

Registered 2026-10-19, before `elo_combo` has produced any prospective forecast.

## Definition (frozen; every component at its already pre-registered setting, no new tuning)
`elo_combo` = the `elo_avail` ridge layer (L2 10, refit each season start on past forecasts, min 200 pairs)
applied on top of a base Elo that combines:
- the tuned Elo (K 20, season regression 0.5, margin of victory on);
- the learned home edge of `elo_hfa` (starts at 48, moves by 1.0 × (outcome − expected) after each
  non-neutral game, clipped to [0, 150]);
- the QB adjustment of `elo_qbadj` (875 × (expected starter rating − season baseline), same expected-starter,
  rating and baseline rules), used in both forecast and rating update.

The availability score uses `elo_avail`'s weights **except the QB group, weighted 0**, because QB absences
are already handled by the QB adjustment. This is the only structural decision; no value was fitted.

## Validation sanity check (2010–2019; not evidence)
`elo_combo` − `elo_tuned` +0.0001 [−0.0018, +0.0020]; vs `elo_hfa` −0.0001 [−0.0020, +0.0017];
vs `elo_avail` +0.0008 [−0.0010, +0.0027]; vs `elo_qbadj` +0.0005 [−0.0004, +0.0014]. The components'
small effects do not visibly add up on validation. Per `docs/power_analysis.md`, effects below ≈ 0.002
are not detectable on these 2,662 games.

## Comparison (pre-specified; same data rules as the other challengers)
- Prospective ledger forecasts for games after the first run including `elo_combo`; latest forecast
  before the actual kickoff; ties excluded.
- Primary metric: paired Brier difference `elo_combo` − `elo_tuned`, week-block bootstrap.
- Multiplicity: now four challengers. `elo_combo` uses a **98.75% interval** (95% Bonferroni-adjusted for
  4). The earlier registrations keep their stated levels (`elo_hfa` 95%; `elo_avail`, `elo_qbadj` 98.3%).
- Decision at the first `sfi score` with ≥ 500 scored games: upper bound < 0 → eligible for promotion;
  lower bound > 0 → retire; otherwise keep running and re-check every 250 games. If several challengers
  are eligible at once, promote the one with the lowest mean Brier difference.
- Expectation, stated in advance: at 500 games the detectable effect is about 0.004–0.006, so "keep
  running" is the most likely outcome.
