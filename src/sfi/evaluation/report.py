"""Score archived forecasts against final results and write a Markdown evaluation report."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sfi.data.schema import Game
from sfi.evaluation import metrics as M
from sfi.forecasts.record import ForecastRecord

ORACLE = "oracle_true_prob"


@dataclass
class ScoredSet:
    model: str
    game_ids: list[str] = field(default_factory=list)
    p: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    block: list[str] = field(default_factory=list)
    n_tie: int = 0
    n_burn_in: int = 0
    n_abstain: int = 0


def score(
    records: list[ForecastRecord], games: dict[str, Game], burn_in_seasons: set[int],
    oracle: dict[str, float] | None = None,
) -> dict[str, ScoredSet]:
    sets: dict[str, ScoredSet] = {}
    for r in records:
        s = sets.setdefault(r.model_name, ScoredSet(r.model_name))
        g = games[r.game_id]
        if r.season in burn_in_seasons:
            s.n_burn_in += 1
        elif r.quality_gate.status != "publish":
            s.n_abstain += 1
        elif g.home_outcome == 0.5:
            s.n_tie += 1
        else:
            s.game_ids.append(g.game_id)
            s.p.append(r.distribution["p_home_win"])
            s.y.append(g.home_outcome)
            s.block.append(f"{g.season}-W{g.week:02d}")
    if oracle is not None and sets:
        ref = next(iter(sets.values()))  # every model forecasts the same games; mirror the first
        o = ScoredSet(ORACLE, n_tie=ref.n_tie, n_burn_in=ref.n_burn_in, n_abstain=ref.n_abstain)
        for gid, y, b in zip(ref.game_ids, ref.y, ref.block):
            o.game_ids.append(gid)
            o.p.append(oracle[gid])
            o.y.append(y)
            o.block.append(b)
        sets[ORACLE] = o
    return sets


def summarize(sets: dict[str, ScoredSet], n_boot: int, seed: int = 0) -> dict[str, dict]:
    out = {}
    for name, s in sets.items():
        if not s.p:
            out[name] = {"n_scored": 0}
            continue
        b, b_lo, b_hi = M.block_bootstrap_mean_ci(M.brier_terms(s.p, s.y), s.block, n_boot, seed=seed)
        ll = M.log_loss(s.p, s.y)
        a, sl = M.calibration_intercept_slope(s.p, s.y)
        acc = float(np.mean((np.asarray(s.p) > 0.5) == (np.asarray(s.y) == 1)))
        out[name] = {
            "n_scored": len(s.p), "n_tie": s.n_tie, "n_abstain": s.n_abstain, "n_burn_in": s.n_burn_in,
            "brier": b, "brier_ci95": [b_lo, b_hi], "log_loss": ll,
            "calibration_intercept": None if np.isnan(a) else a,
            "calibration_slope": None if np.isnan(sl) else sl, "accuracy_at_0.5": acc,
        }
    return out


def paired_brier_diff(sets: dict[str, ScoredSet], a: str, b: str, n_boot: int, seed: int = 0):
    """Brier(a) - Brier(b) on the same games, CI by week-block bootstrap. Negative favors a."""
    sa, sb = sets[a], sets[b]
    if sa.game_ids != sb.game_ids:
        raise ValueError("paired comparison requires identical scored games")
    d = M.brier_terms(sa.p, sa.y) - M.brier_terms(sb.p, sb.y)
    return M.block_bootstrap_mean_ci(d, sa.block, n_boot, seed=seed)


def write_markdown(
    path: str | Path, *, title: str, data_kind: str, snapshot_id: str, config_note: str,
    summary: dict[str, dict], sets: dict[str, ScoredSet], primary: str,
    comparisons: list[tuple[str, str]], n_boot: int, n_bins: int,
) -> None:
    lines = [f"# {title}", ""]
    if data_kind == "synthetic":
        lines += ["> **SYNTHETIC DATA — NOT REAL NFL RESULTS.** Teams, games and scores are generated",
                  "> by `sfi.data.synthetic`. These numbers test the pipeline; they say nothing about",
                  "> real-world forecasting skill.", ""]
    lines += [f"- Data snapshot: `{snapshot_id}`", f"- {config_note}",
              "- Outcome: P(home win), ties excluded from scoring and counted separately.",
              "- Split: walk-forward by week; forecasts made 1h before each week's first kickoff "
              "using only results known by then.", ""]

    lines += ["## Scores (lower is better for Brier and log loss)", "",
              "| Model | Scored | Brier | Brier 95% CI | Log loss | Calib. intercept | Calib. slope | Acc @0.5 | Ties | Abstained | Burn-in |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summary.items():
        if s.get("n_scored", 0) == 0:
            lines.append(f"| {name} | 0 | – | – | – | – | – | – | – | – | – |")
            continue
        lo, hi = s["brier_ci95"]
        a, sl = s["calibration_intercept"], s["calibration_slope"]
        a_txt, sl_txt = ("n/a", "n/a") if sl is None else (f"{a:+.3f}", f"{sl:.3f}")
        lines.append(
            f"| {name} | {s['n_scored']} | {s['brier']:.4f} | [{lo:.4f}, {hi:.4f}] | {s['log_loss']:.4f} | "
            f"{a_txt} | {sl_txt} | {s['accuracy_at_0.5']:.3f} | "
            f"{s['n_tie']} | {s['n_abstain']} | {s['n_burn_in']} |")
    if ORACLE in summary:
        lines += ["", "Calibration slope/intercept are n/a for near-constant forecasts (not identifiable). "
                  "A slope below 1 means forecasts are too extreme; above 1, too timid."]
        lines += ["", f"`{ORACLE}` uses the generator's hidden true probabilities. It is the best any model "
                  "could do on this synthetic data and is **not available for real data**."]

    lines += ["", "## Paired comparisons (Brier difference, week-block bootstrap)", "",
              "| A − B | Mean diff | 95% CI | Reading |", "|---|---:|---|---|"]
    for a, b in comparisons:
        if a not in sets or b not in sets or not sets[a].p:
            continue
        m, lo, hi = paired_brier_diff(sets, a, b, n_boot)
        reading = ("A better" if hi < 0 else "B better" if lo > 0 else "no clear difference")
        lines.append(f"| {a} − {b} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {reading} |")

    if primary in sets and sets[primary].p:
        s = sets[primary]
        lines += ["", f"## Reliability table: `{primary}`", "",
                  "| Forecast bin | Games | Mean forecast | Observed home-win rate |", "|---|---:|---:|---:|"]
        for c in M.calibration_table(s.p, s.y, n_bins):
            lines.append(f"| {c.lo:.1f}–{c.hi:.1f} | {c.n} | {c.mean_p:.3f} | {c.observed:.3f} |")
        lines += ["", "Bins with few games are noisy; do not read calibration from small counts."]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
