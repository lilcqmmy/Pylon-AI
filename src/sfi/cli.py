"""Command line.

  sfi synth     generate fictional NFL-like seasons
  sfi tune      grid-search Elo on validation seasons -> artifacts/elo_tuned.json
  sfi backtest  walk-forward forecasts for all models, scored separately on validation and holdout
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from sfi.config import PROJECT_ROOT, load_config
from sfi.data import synthetic
from sfi.data.schema import file_sha256, read_games_csv, write_games_csv
from sfi.evaluation.report import ORACLE, score, section_lines, summarize, write_markdown
from sfi.evaluation.tuning import tune_elo
from sfi.evaluation.walkforward import GateConfig, run_walkforward
from sfi.forecasts.record import ForecastArchive
from sfi.models.baselines import CoinFlip, HomeRate
from sfi.models.calibration import PlattCalibrated
from sfi.models.elo import EloModel, EloParams

SYN_DIR = PROJECT_ROOT / "data" / "synthetic"
GAMES_CSV = SYN_DIR / "games.csv"
TRUTH_CSV = SYN_DIR / "truth_DO_NOT_USE_AS_FEATURE.csv"


def splits(cfg: dict, available: set[int]) -> tuple[set[int], set[int], set[int]]:
    v = cfg["validation"]
    burn, val, hold = (set(v[k]) for k in ("burn_in_seasons", "validation_seasons", "holdout_seasons"))
    if not val or not hold:
        raise SystemExit("config: validation_seasons and holdout_seasons must be non-empty")
    if (burn & val) or (burn & hold) or (val & hold):
        raise SystemExit("config: burn-in, validation and holdout seasons must be disjoint")
    if (burn and max(burn) >= min(val)) or max(val) >= min(hold):
        raise SystemExit("config: seasons must be ordered burn-in < validation < holdout")
    missing = (burn | val | hold) - available
    if missing:
        raise SystemExit(f"config: seasons {sorted(missing)} not in data; re-run `sfi synth`?")
    return burn, val, hold


def load_games():
    if not GAMES_CSV.exists():
        print("No synthetic data yet; run `sfi synth` first.", file=sys.stderr)
        raise SystemExit(1)
    return read_games_csv(GAMES_CSV), "sha256:" + file_sha256(GAMES_CSV)[:16]


def default_params(cfg: dict) -> EloParams:
    return EloParams(**{k: cfg["elo"][k] for k in EloParams.__dataclass_fields__})


def cmd_synth(cfg: dict) -> int:
    s = cfg["synthetic"]
    games, truth = synthetic.generate(s["seed"], s["seasons"], s["n_teams"], s["weeks_per_season"])
    write_games_csv(games, GAMES_CSV)
    synthetic.write_truth_csv(truth, TRUTH_CSV)
    print(f"Wrote {len(games)} synthetic games to {GAMES_CSV.relative_to(PROJECT_ROOT)}")
    print(f"sha256 {file_sha256(GAMES_CSV)}")
    return 0


def cmd_tune(cfg: dict) -> int:
    games, snapshot_id = load_games()
    _, val, hold = splits(cfg, {g.season for g in games})
    t = cfg["tuning"]
    trials = tune_elo(games, t["grid"], val, base=default_params(cfg), objective=t["objective"])
    best = trials[0]
    default = next((x for x in trials if x.params == default_params(cfg)), None)
    out = {
        "data_kind": "synthetic", "snapshot_id": snapshot_id, "objective": t["objective"],
        "validation_seasons": sorted(val), "holdout_seasons_untouched": sorted(hold),
        "n_trials": len(trials), "grid": t["grid"],
        "best": {"params": asdict(best.params), "brier": best.brier, "log_loss": best.log_loss, "n": best.n},
        "top10": [{"params": asdict(x.params), "brier": x.brier, "log_loss": x.log_loss} for x in trials[:10]],
    }
    path = PROJECT_ROOT / t["artifact"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{len(trials)} configurations scored on validation seasons {sorted(val)} "
          f"({best.n} games); holdout {sorted(hold)} not loaded.")
    print(f"Best {t['objective']}: Brier={best.brier:.4f} LogLoss={best.log_loss:.4f} params={asdict(best.params)}")
    if default is None:
        print("(default params are not in the grid)")
    else:
        print(f"Default params:    Brier={default.brier:.4f} LogLoss={default.log_loss:.4f}")
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")
    return 0


def cmd_backtest(cfg: dict) -> int:
    games, snapshot_id = load_games()
    _, val, hold = splits(cfg, {g.season for g in games})
    tuned_path = PROJECT_ROOT / cfg["tuning"]["artifact"]
    if not tuned_path.exists():
        print("No tuned Elo yet; run `sfi tune` first.", file=sys.stderr)
        return 1
    tuned = json.loads(tuned_path.read_text(encoding="utf-8"))
    if tuned["snapshot_id"] != snapshot_id:
        print("Tuned Elo was fit on a different data snapshot; re-run `sfi tune`.", file=sys.stderr)
        return 1
    if sorted(val) != tuned["validation_seasons"]:
        print("Validation seasons changed since tuning; re-run `sfi tune`.", file=sys.stderr)
        return 1
    tuned_params = EloParams(**tuned["best"]["params"])
    default = default_params(cfg)
    c = cfg["calibration"]
    gate = GateConfig(cfg["forecast"]["publish_gate"].get("min_prior_games_per_team") or 0)

    factories = [
        CoinFlip, HomeRate,
        lambda: EloModel(default, name="elo_default"),
        lambda: EloModel(tuned_params, name="elo_tuned"),
        lambda: PlattCalibrated(EloModel(tuned_params, name="elo_tuned"), min_samples=c["min_samples"],
                                exclude_first_season=c["exclude_first_season"], name="elo_tuned_platt"),
    ]
    records = run_walkforward(games, factories, source_snapshot_id=snapshot_id,
                              data_kind="synthetic", gate=gate)
    archive = ForecastArchive(PROJECT_ROOT / cfg["storage"]["backtest_archive"])
    written, same = archive.append(records)
    print(f"Archive: {written} new records, {same} identical re-runs ({len(archive)} total)")

    v = cfg["validation"]
    by_id = {g.game_id: g for g in games}
    truth = {gid: t.p_home_win_given_no_tie for gid, t in synthetic.read_truth_csv(TRUTH_CSV).items()}
    results, sections = {}, []
    split_defs = [
        ("validation", val, f"Validation seasons {sorted(val)}",
         "Elo parameters were **selected on these seasons**, so `elo_tuned` scores here are optimistic."),
        ("holdout", hold, f"Holdout seasons {sorted(hold)}",
         "Never used for tuning or any design choice. This is the honest estimate."),
    ]
    for key, seasons, title, note in split_defs:
        sets = score(records, by_id, seasons, oracle=truth)
        summary = summarize(sets, v["bootstrap_samples"])
        results[key] = summary
        sections.append(section_lines(
            title, note, sets, summary,
            comparisons=[("elo_tuned", "elo_default"), ("elo_tuned_platt", "elo_tuned"),
                         ("elo_tuned", "home_rate"), ("elo_tuned_platt", ORACLE)],
            reliability=["elo_tuned", "elo_tuned_platt"] if key == "holdout" else [],
            n_boot=v["bootstrap_samples"], n_bins=v["calibration_bins"],
        ))

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "baseline_synthetic.json").write_text(json.dumps(
        {"data_kind": "synthetic", "snapshot_id": snapshot_id, "splits": {k: sorted(s) for k, s, *_ in split_defs},
         "elo_default": asdict(default), "elo_tuned": asdict(tuned_params), "calibration": c,
         "results": results}, indent=2), encoding="utf-8")
    write_markdown(
        reports / "baseline_synthetic.md", title="Walk-forward evaluation (synthetic)",
        data_kind="synthetic",
        header=[f"Data snapshot: `{snapshot_id}`",
                f"Burn-in (not scored): {v['burn_in_seasons']}; validation: {sorted(val)}; holdout: {sorted(hold)}",
                f"`elo_default`: {asdict(default)}",
                f"`elo_tuned`: {asdict(tuned_params)} (best of {tuned['n_trials']} by validation {tuned['objective']})",
                f"`elo_tuned_platt`: Platt scaling refit before each week on past issued forecasts only "
                f"(min {c['min_samples']} samples; first season excluded: {c['exclude_first_season']})",
                "Outcome: P(home win); ties excluded from scoring and counted separately.",
                "Forecasts made 1h before each week's first kickoff using only results known by then."],
        sections=sections,
    )
    for key, summary in results.items():
        print(f"[{key}]")
        for name, s in summary.items():
            if s.get("n_scored"):
                print(f"  {name:18s} n={s['n_scored']:4d}  Brier={s['brier']:.4f}  LogLoss={s['log_loss']:.4f}")
    print("Report: reports/baseline_synthetic.md  (SYNTHETIC — not real NFL results)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sfi", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to YAML config (default: config/config.yaml or example)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("synth", help="generate fictional NFL-like seasons")
    sub.add_parser("tune", help="grid-search Elo on validation seasons")
    sub.add_parser("backtest", help="walk-forward backtest + validation/holdout report")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    return {"synth": cmd_synth, "tune": cmd_tune, "backtest": cmd_backtest}[args.cmd](cfg)


if __name__ == "__main__":
    raise SystemExit(main())
