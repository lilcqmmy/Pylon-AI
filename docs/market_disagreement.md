# Market-disagreement study, 2006–2019 (hypothesis generator only)

Computed 2026-10-19 with `scripts/analysis/market_disagreement.py` (validation seasons only; holdout not
used). Games bucketed by d = `elo_tuned` P(home) − market P(home), where the market probability is the
de-vigged moneyline in the nflverse snapshot. **The line's recording time is undocumented (likely near
kickoff), and this is retrospective**, so nothing here is evidence of an edge; the prospective
line-movement ledger (`ledger/market_moves.jsonl`) and the graded paper picks are the real test.
Paper return: 1 unit on the side Elo prefers (home if d > 0) at the snapshot moneyline; ties push;
95% week-block bootstrap intervals. 3,602 games with both moneylines (2006–2019; 2006 and 2008 partial).

| d = Elo − market | Games | Elo Brier | Market Brier | Paper return per unit [95% CI] |
|---|---:|---:|---:|---|
| ≤ −0.15 | 199 | 0.2416 | 0.2134 | +0.099 [−0.154, +0.351] |
| −0.15 to −0.10 | 353 | 0.2116 | 0.2032 | +0.042 [−0.124, +0.212] |
| −0.10 to −0.05 | 688 | 0.2081 | 0.2039 | −0.012 [−0.128, +0.106] |
| −0.05 to 0 | 853 | 0.2040 | 0.2038 | −0.041 [−0.126, +0.045] |
| 0 to 0.05 | 695 | 0.2232 | 0.2216 | −0.028 [−0.100, +0.048] |
| 0.05 to 0.10 | 446 | 0.2238 | 0.2218 | +0.019 [−0.096, +0.140] |
| 0.10 to 0.15 | 227 | 0.2385 | 0.2346 | +0.115 [−0.066, +0.302] |
| ≥ 0.15 | 141 | 0.2718 | 0.1978 | −0.196 [−0.421, +0.050] |

Pooled paper return at the live paper-pick thresholds (|d| ≥ x):

| Threshold | Games | Return per unit [95% CI] |
|---|---:|---|
| 0.03 | 2,656 | −0.001 [−0.058, +0.056] |
| 0.05 | 2,054 | +0.016 [−0.051, +0.080] |
| 0.10 | 920 | +0.036 [−0.068, +0.144] |

## Reading
- The market has the lower Brier score in **every** bucket, and its advantage grows with disagreement: when
  Elo is 0.15+ more bullish on the home team, Elo scores 0.272 vs the market's 0.198. Large disagreements
  are mostly Elo errors (missing injuries, QB changes, news).
- Paper returns are indistinguishable from zero in every bucket and pooled; the intervals are ±5–25% per
  unit, so even this 14-season sample cannot separate a small edge from noise.
- Hypotheses worth tracking prospectively (not acting on): whether moderate disagreements (0.05–0.15) show
  positive closing-line value in the live ledger, and whether the market moves toward Elo's side when
  they disagree.
