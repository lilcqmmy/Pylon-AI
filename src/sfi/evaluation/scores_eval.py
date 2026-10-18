"""Walk-forward margin/total forecasts and their evaluation (CRPS, MAE, RMSE, pinball, interval
coverage with Wilson intervals, randomized PIT)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterator

import numpy as np

from sfi.data.schema import Game
from sfi.distributions import DiscreteDist, wilson_interval
from sfi.evaluation import metrics as M
from sfi.evaluation.walkforward import Replay, slates_by_kickoff
from sfi.forecasts.record import ForecastRecord, QualityGate, _iso

LEVELS = (0.5, 0.8, 0.9)
PINBALL_QS = (0.1, 0.25, 0.5, 0.75, 0.9)
CHI2_DF9_95 = 16.919  # 95th percentile of chi-square with 9 degrees of freedom (PIT deciles)


@dataclass
class ScoreForecast:
    game: Game
    as_of: datetime
    margin: DiscreteDist
    total: DiscreteDist
    inputs: dict


def iter_score_walkforward(games: list[Game], model, *, as_of_lead: timedelta = timedelta(hours=1)) -> Iterator[ScoreForecast]:
    replay = Replay(games, model)
    for kickoff, slate in slates_by_kickoff(games):
        replay.advance_to(kickoff - as_of_lead)
        for g in slate:
            replay.prepare(g)
            m, t = model.predict_scores(g)
            yield ScoreForecast(g, replay.as_of, m, t, model.inputs_for(g))


def pmf_to_json(d: DiscreteDist, digits: int = 6) -> dict:
    """Compact pmf: trimmed to the range with mass, rounded. Reload with `pmf_from_json`."""
    p = np.round(d.pmf, digits)
    nz = np.nonzero(p)[0]
    if len(nz) == 0:
        return {"lo": d.lo, "p": [1.0]}
    a, b = int(nz[0]), int(nz[-1])
    return {"lo": d.lo + a, "p": [float(x) for x in p[a:b + 1]]}


def pmf_from_json(obj: dict) -> DiscreteDist:
    return DiscreteDist(int(obj["lo"]), np.asarray(obj["p"], dtype=float))


def make_score_record(f: ScoreForecast, model, *, source_snapshot_id: str, data_kind: str,
                      include_pmf: bool = False) -> ForecastRecord:
    """`include_pmf` stores the full distributions (used for live forecasts so they can be scored exactly
    later); backtest archives keep quantile summaries only."""
    import hashlib
    from dataclasses import asdict

    if f.as_of >= f.game.scheduled_start_utc:
        raise ValueError(f"{f.game.game_id}: forecast as_of is not before kickoff")
    stamp = f.as_of.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap = source_snapshot_id.split(":")[-1][:12]
    g = f.game
    rec = ForecastRecord(
        forecast_id=f"{model.name}-{model.version}-{snap}-{g.game_id}-{stamp}-margin_total",
        league=g.league, season=g.season, week=g.week, game_id=g.game_id, home_team=g.home_team,
        away_team=g.away_team, outcome_type="home_margin_and_total", prediction_as_of_utc=_iso(f.as_of),
        scheduled_start_utc=_iso(g.scheduled_start_utc), source_snapshot_id=source_snapshot_id,
        feature_spec_version=model.feature_spec_version, model_name=model.name, model_version=model.version,
        distribution={"margin": f.margin.summary(), "total": f.total.summary(),
                      "p_home_win_implied": round(implied_p_home(f.margin), 6),
                      **({"margin_pmf": pmf_to_json(f.margin), "total_pmf": pmf_to_json(f.total)} if include_pmf else {})},
        inputs=f.inputs, quality_gate=QualityGate("publish"), data_kind=data_kind,
        simulation_only=data_kind == "synthetic",
    )
    digest = hashlib.sha256(rec.canonical_json().encode()).hexdigest()
    return ForecastRecord(**{**asdict(rec), "quality_gate": rec.quality_gate, "record_hash": digest})


def implied_p_home(margin: DiscreteDist) -> float:
    """P(home wins | no tie) implied by the margin distribution."""
    pw, pl = margin.p_greater(0), margin.cdf_at(-1)
    return pw / (pw + pl) if pw + pl > 0 else 0.5


@dataclass
class TargetEval:
    model: str
    game_ids: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    y: list[int] = field(default_factory=list)
    crps: list[float] = field(default_factory=list)
    abs_err_median: list[float] = field(default_factory=list)
    sq_err_mean: list[float] = field(default_factory=list)
    pinball: list[float] = field(default_factory=list)
    hits: dict = field(default_factory=lambda: {lv: [] for lv in LEVELS})
    claimed: dict = field(default_factory=lambda: {lv: [] for lv in LEVELS})
    width: dict = field(default_factory=lambda: {lv: [] for lv in LEVELS})
    pit: list[float] = field(default_factory=list)


def evaluate(forecasts: list[tuple[str, ScoreForecast]], target: str, seasons: set[int], seed: int = 0) -> dict[str, TargetEval]:
    """forecasts: [(model_name, forecast)]. target: 'margin' or 'total'."""
    rng = np.random.default_rng(seed)
    out: dict[str, TargetEval] = {}
    for name, f in forecasts:
        g = f.game
        if g.season not in seasons or not g.is_final:
            continue
        y = g.home_score - g.away_score if target == "margin" else g.home_score + g.away_score
        d = f.margin if target == "margin" else f.total
        e = out.setdefault(name, TargetEval(name))
        e.game_ids.append(g.game_id)
        e.blocks.append(f"{g.season}-W{g.week:02d}")
        e.y.append(y)
        e.crps.append(d.crps(y))
        e.abs_err_median.append(abs(y - d.quantile(0.5)))
        e.sq_err_mean.append((y - d.mean()) ** 2)
        e.pinball.append(d.pinball(y, PINBALL_QS))
        for lv in LEVELS:
            a, b = d.central_interval(lv)
            e.hits[lv].append(float(a <= y <= b))
            e.claimed[lv].append(d.mass(a, b))
            e.width[lv].append(b - a)
        e.pit.append(d.pit(y, float(rng.uniform())))
    return out


def summarize_target(evals: dict[str, TargetEval], n_boot: int) -> dict[str, dict]:
    res = {}
    for name, e in evals.items():
        crps, lo, hi = M.block_bootstrap_mean_ci(e.crps, e.blocks, n_boot)
        cov = {}
        for lv in LEVELS:
            h = int(sum(e.hits[lv]))
            wl, wh = wilson_interval(h, len(e.hits[lv]))
            claimed = float(np.mean(e.claimed[lv]))
            cov[str(lv)] = {"observed": h / len(e.hits[lv]), "claimed": claimed, "wilson95": [wl, wh],
                            "consistent": wl <= claimed <= wh, "mean_width": float(np.mean(e.width[lv]))}
        counts = np.histogram(e.pit, bins=10, range=(0, 1))[0]
        exp = len(e.pit) / 10
        chi2 = float(((counts - exp) ** 2 / exp).sum())
        res[name] = {"n": len(e.y), "crps": crps, "crps_ci95": [lo, hi], "mae_median": float(np.mean(e.abs_err_median)),
                     "rmse_mean": float(np.sqrt(np.mean(e.sq_err_mean))), "pinball": float(np.mean(e.pinball)),
                     "coverage": cov, "pit_deciles": counts.tolist(), "pit_chi2": chi2,
                     "pit_uniform_at_5pct": chi2 <= CHI2_DF9_95}
    return res


def paired_diff(a: TargetEval, b: TargetEval, attr: str, n_boot: int):
    ib = {gid: i for i, gid in enumerate(b.game_ids)}
    pairs = [(i, ib[g]) for i, g in enumerate(a.game_ids) if g in ib]
    va, vb = getattr(a, attr), getattr(b, attr)
    d = [va[i] - vb[j] for i, j in pairs]
    m, lo, hi = M.block_bootstrap_mean_ci(d, [a.blocks[i] for i, _ in pairs], n_boot)
    return m, lo, hi, len(pairs)


def market_point_eval(target: str, lines: dict[str, dict], games: dict[str, Game], seasons: set[int]) -> TargetEval:
    """Market spread/total as a point forecast (benchmark only): absolute and squared errors."""
    key = "spread" if target == "margin" else "total"
    e = TargetEval(f"market_{key}_line")
    for gid, ln in lines.items():
        g = games.get(gid)
        if g is None or g.season not in seasons or not g.is_final or ln.get(key) is None:
            continue
        y = g.home_score - g.away_score if target == "margin" else g.home_score + g.away_score
        x = ln[key]
        e.game_ids.append(gid)
        e.blocks.append(f"{g.season}-W{g.week:02d}")
        e.y.append(y)
        e.abs_err_median.append(abs(y - x))
        e.sq_err_mean.append((y - x) ** 2)
    return e


def section_lines(target: str, title: str, note: str, evals: dict[str, TargetEval], summary: dict[str, dict],
                  market: TargetEval | None, comparisons: list[tuple[str, str]], n_boot: int) -> list[str]:
    unit = "points (home − away)" if target == "margin" else "total points"
    lines = [f"### {title}: {target}", "", note, "",
             f"| Model | Games | CRPS | CRPS 95% CI | MAE (median) | RMSE (mean) | Pinball | PIT χ² (9 df) |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    for name, s in summary.items():
        flag = "" if s["pit_uniform_at_5pct"] else " ⚠"
        lines.append(f"| {name} | {s['n']} | {s['crps']:.3f} | [{s['crps_ci95'][0]:.3f}, {s['crps_ci95'][1]:.3f}] | "
                     f"{s['mae_median']:.2f} | {s['rmse_mean']:.2f} | {s['pinball']:.3f} | {s['pit_chi2']:.1f}{flag} |")
    if market is not None and market.y:
        lines.append(f"| {market.model} (benchmark, point only) | {len(market.y)} | – | – | "
                     f"{np.mean(market.abs_err_median):.2f} | {np.sqrt(np.mean(market.sq_err_mean)):.2f} | – | – |")
    lines += ["", f"Units: {unit}. CRPS, MAE, RMSE and pinball: lower is better. PIT χ² above "
              f"{CHI2_DF9_95} (⚠) means the predictive distribution's shape is detectably miscalibrated.", "",
              "Central-interval coverage (observed hit rate vs the forecast's own claimed mass, with 95% Wilson "
              "interval; ✓ when the claimed mass lies inside it):", "",
              "| Model | 50%: observed / claimed | 80%: observed / claimed | 90%: observed / claimed | Mean width 50/80/90 |",
              "|---|---|---|---|---|"]
    for name, s in summary.items():
        cells = []
        for lv in LEVELS:
            c = s["coverage"][str(lv)]
            mark = "✓" if c["consistent"] else "✗"
            cells.append(f"{c['observed']:.3f} / {c['claimed']:.3f} [{c['wilson95'][0]:.3f}, {c['wilson95'][1]:.3f}] {mark}")
        widths = "/".join(f"{s['coverage'][str(lv)]['mean_width']:.1f}" for lv in LEVELS)
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {widths} |")
    lines += ["", "Paired differences on shared games (week-block bootstrap; negative favors A):", "",
              "| A − B | Metric | Games | Mean diff | 95% CI | Reading |", "|---|---|---:|---:|---|---|"]
    for a, b in comparisons:
        ea = evals.get(a) or (market if market is not None and market.model == a else None)
        eb = evals.get(b) or (market if market is not None and market.model == b else None)
        if ea is None or eb is None or not ea.y or not eb.y:
            continue
        metrics = ["crps"] if (ea.crps and eb.crps) else []
        metrics += ["abs_err_median"]
        for attr in metrics:
            m, lo, hi, n = paired_diff(ea, eb, attr, n_boot)
            reading = "A better" if hi < 0 else "B better" if lo > 0 else "no clear difference"
            label = {"crps": "CRPS", "abs_err_median": "abs. error"}[attr]
            lines.append(f"| {a} − {b} | {label} | {n} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {reading} |")
    return lines + [""]
