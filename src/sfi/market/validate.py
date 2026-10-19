"""Validation run for the market-residual family (2006-2017 only; holdout 2018-2025 sealed)."""

from __future__ import annotations

import math

import numpy as np

from sfi.evaluation.metrics import block_bootstrap_mean_ci
from sfi.market import hypotheses as H
from sfi.market.dataset import MarketRow
from sfi.market.model import TARGETS, bootstrap_coefficients, market_calibration, names, walk_forward

LABEL = ("Market lines are treated as the close (nflverse records one line per game; timing undocumented). "
         "Market-residual family: separate from the independent forecast, never blended. Research only.")


def _pct(x):
    return f"{100 * x:.1f}%"


def hypothesis_section(results: list[dict]) -> list[str]:
    lines = ["## Pre-registered hypotheses (validation 2006–2017)", "",
             "Exact two-sided binomial test vs 50% (Clopper–Pearson 95% CI); difference hypotheses: two-proportion z-test. "
             "Holm correction across the 10-test family (e2 dropped, Amendment 1). MDE = smallest true deviation from 50% (or difference) detectable "
             "with 80% power at 95%. Break-even at −110 is 52.38%. Return: 1 unit per game at the recorded prices.", "",
             "| ID | Hypothesis | Games | Rate / difference | 95% CI | MDE | p | Holm p | Paper return per unit [95% CI] | Supported |",
             "|---|---|---:|---:|---|---:|---:|---:|---|---|"]
    for r in results:
        if r["kind"] == "rate":
            est = _pct(r["estimate"])
            ci = f"[{_pct(r['ci'][0])}, {_pct(r['ci'][1])}]"
            n = f"{r['n']} (+{r['pushes']} pushes)"
            ret = f"{r['return_per_unit']:+.3f} [{r['return_ci'][0]:+.3f}, {r['return_ci'][1]:+.3f}]"
            mde = f"±{100 * r['mde']:.1f} pts"
        else:
            est = f"{100 * r['estimate']:+.1f} pts ({_pct(r['rate1'])} vs {_pct(r['rate2'])})"
            ci = f"[{100 * r['ci'][0]:+.1f}, {100 * r['ci'][1]:+.1f}] pts"
            n = f"{r['n1']} vs {r['n2']}"
            ret = "–"
            mde = f"±{100 * r['mde']:.1f} pts"
        lines.append(f"| {r['id']} | {r['description']} | {n} | {est} | {ci} | {mde} | {r['p']:.3f} | {r['p_holm']:.3f} | "
                     f"{ret} | {'**yes**' if r['supported'] else 'no'} |")
    sec = []
    for r in results:
        if "secondary" in r:
            s = r["secondary"]
            if r["id"] == "d":
                sec.append(f"- d (secondary): divisional underdog cover {_pct(s['div_underdog_rate'])} "
                           f"[{_pct(s['ci'][0])}, {_pct(s['ci'][1])}], p vs 50% = {s['p_vs_50']:.3f}.")
            if r["id"] == "f":
                sec.append(f"- f (secondary): non-key underdog cover {_pct(s['non_key_underdog_rate'])} (n {s['n_non_key']}); "
                           f"key minus non-key {100 * s['key_minus_non_key']:+.1f} pts "
                           f"[{100 * s['ci'][0]:+.1f}, {100 * s['ci'][1]:+.1f}], p = {s['p']:.3f}.")
    supported = [r["id"] for r in results if r["supported"]]
    lines += ["", *sec, "", f"Supported in validation (Holm p < 0.05): {', '.join(supported) if supported else 'none'}. "
              "Only these would be tested on the holdout, which stays sealed.", ""]
    return lines


def model_section(rows: list[MarketRow], seasons: list[int], l2: float, n_boot: int) -> list[str]:
    lines = ["## market_logit (validation 2006–2017)", "",
             f"Logistic regression on the market logit plus indicators; ridge L2 {l2:g} on the indicators only. "
             "Calibration and coefficients: fit on all of 2006–2017, 95% CIs from resampling whole weeks. "
             "Paired Brier: walk-forward (each season from a fit on earlier seasons; first two seasons are training only).", ""]
    for target in TARGETS:
        label = {"win": "home win (market = de-vigged moneyline)", "cover": "home ATS cover (market = de-vigged spread prices)",
                 "over": "over (market = de-vigged total prices)"}[target]
        cal, clo, chi, n = market_calibration(rows, target, n_boot)
        beta, lo, hi, _ = bootstrap_coefficients(rows, target, l2, n_boot)
        wf = walk_forward(rows, target, seasons, l2)
        d = (wf.p_model - wf.y) ** 2 - (wf.p_market - wf.y) ** 2
        m, dlo, dhi = block_bootstrap_mean_ci(d, wf.blocks, n_boot * 5)
        se = (dhi - dlo) / (2 * 1.959964)
        mde = (1.959964 + 0.841621) * se
        lines += [f"### Target: {label}", "",
                  f"- Market alone (n = {n}): intercept {cal[0]:+.3f} [{clo[0]:+.3f}, {chi[0]:+.3f}], "
                  f"slope {cal[1]:.3f} [{clo[1]:.3f}, {chi[1]:.3f}] (perfect: 0 and 1).",
                  f"- Walk-forward {wf.rows[0].game.season if wf.rows else '–'}–{seasons[-1]} (n = {len(wf.y)}): "
                  f"Brier market_logit {np.mean((wf.p_model - wf.y) ** 2):.4f} vs market alone "
                  f"{np.mean((wf.p_market - wf.y) ** 2):.4f}; paired difference **{m:+.4f} [{dlo:+.4f}, {dhi:+.4f}]**, "
                  f"MDE {mde:.4f}.", "",
                  "| Term | Coefficient | 95% CI |", "|---|---:|---|",
                  f"| intercept | {beta[0]:+.3f} | [{lo[0]:+.3f}, {hi[0]:+.3f}] |",
                  f"| market logit | {beta[1]:+.3f} | [{lo[1]:+.3f}, {hi[1]:+.3f}] |"]
        for i, nm in enumerate(names(target)):
            lines.append(f"| {nm} | {beta[2 + i]:+.3f} | [{lo[2 + i]:+.3f}, {hi[2 + i]:+.3f}] |")
        lines.append("")
    return lines


def mde_note() -> list[str]:
    ex = {n: 100 * H.mde_rate(n) for n in (300, 1000, 3000)}
    return ["## Reading \"no effect\"", "",
            f"For a single ATS rate, the 80%-power MDE at 95% is about ±{ex[300]:.1f} points with 300 games, "
            f"±{ex[1000]:.1f} with 1,000 and ±{ex[3000]:.1f} with 3,000. The margin needed just to beat the vig at −110 is "
            "2.4 points above 50%, so only the largest subsets can resolve a bias of that size, and the Holm correction "
            "raises the bar further. A non-significant result means \"not detectable here\", not \"absent\".", ""]
