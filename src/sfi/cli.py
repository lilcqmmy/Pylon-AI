"""Command line.

  sfi synth                       generate fictional NFL-like seasons
  sfi fetch                       download a new nflverse schedules snapshot (network)
  sfi tune     [--dataset NAME]   grid-search Elo on validation seasons
  sfi backtest [--dataset NAME] [--reveal-holdout]
                                  walk-forward forecasts for all models; holdout stays sealed unless
                                  --reveal-holdout is given (each reveal is logged)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from sfi.config import PROJECT_ROOT, load_config
from sfi.data import synthetic
from sfi.data.schema import file_sha256, write_games_csv
from sfi.data.sources import nflverse
from sfi.datasets import SYN_GAMES, SYN_TRUTH, DatasetError, load_dataset
from sfi.evaluation.report import score, section_lines, summarize, write_markdown
from sfi.evaluation.tuning import tune_elo
from sfi.evaluation.walkforward import GateConfig, run_walkforward
from sfi.forecasts.record import ForecastArchive
from sfi.models.baselines import CoinFlip, HomeRate
from sfi.models.calibration import PlattCalibrated
from sfi.models.elo import EloModel, EloParams


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_params(cfg: dict) -> EloParams:
    return EloParams(**{k: cfg["elo"][k] for k in EloParams.__dataclass_fields__})


def tuned_path(cfg: dict, dataset: str):
    return PROJECT_ROOT / cfg["tuning"]["artifact_dir"] / f"{dataset}_elo_tuned.json"


def cmd_synth(cfg: dict, args) -> int:
    s = cfg["datasets"]["synthetic"]
    games, truth = synthetic.generate(s["seed"], s["seasons"], s["n_teams"], s["weeks_per_season"])
    write_games_csv(games, SYN_GAMES)
    synthetic.write_truth_csv(truth, SYN_TRUTH)
    print(f"Wrote {len(games)} synthetic games to {SYN_GAMES.relative_to(PROJECT_ROOT)}")
    print(f"sha256 {file_sha256(SYN_GAMES)}")
    return 0


def cmd_fetch(cfg: dict, args) -> int:
    root = PROJECT_ROOT / cfg["datasets"]["nflverse"]["raw_dir"]
    folder, manifest, created = nflverse.fetch_schedules(root)
    state = "New snapshot" if created else "Unchanged since latest snapshot"
    print(f"{state}: {folder.relative_to(PROJECT_ROOT)}  ({manifest['bytes']} bytes, "
          f"sha256 {manifest['sha256'][:16]}, upstream updated {manifest['upstream_last_updated']})")
    print(manifest["attribution"])
    return 0


def cmd_tune(cfg: dict, args) -> int:
    ds = load_dataset(cfg, args.dataset)
    val, hold = ds.splits.validation, ds.splits.holdout
    t = cfg["tuning"]
    games = ds.games_through(max(val))  # holdout seasons are never loaded
    trials = tune_elo(games, t["grid"], set(val), base=default_params(cfg), objective=t["objective"])
    best = trials[0]
    default = next((x for x in trials if x.params == default_params(cfg)), None)
    out = {
        "dataset": ds.name, "data_kind": ds.data_kind, "snapshot_id": ds.snapshot_id,
        "objective": t["objective"], "validation_seasons": sorted(val),
        "holdout_seasons_untouched": sorted(hold), "n_trials": len(trials), "grid": t["grid"],
        "created_at_utc": _now(),
        "best": {"params": asdict(best.params), "brier": best.brier, "log_loss": best.log_loss, "n": best.n},
        "top10": [{"params": asdict(x.params), "brier": x.brier, "log_loss": x.log_loss} for x in trials[:10]],
    }
    edge = [k for k, vals in t["grid"].items()
            if len(vals) > 2 and not isinstance(vals[0], bool) and getattr(best.params, k) in (min(vals), max(vals))]
    path = tuned_path(cfg, ds.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[{ds.name}] {len(trials)} configurations scored on validation {min(val)}-{max(val)} "
          f"({best.n} games); holdout {min(hold)}-{max(hold)} not loaded.")
    print(f"Best {t['objective']}: Brier={best.brier:.4f} LogLoss={best.log_loss:.4f} params={asdict(best.params)}")
    if default is not None:
        print(f"Default params:    Brier={default.brier:.4f} LogLoss={default.log_loss:.4f}")
    if edge:
        print(f"WARNING: best value is at the edge of the grid for {edge}; consider widening it.")
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")
    return 0


def cmd_backtest(cfg: dict, args) -> int:
    ds = load_dataset(cfg, args.dataset)
    val, hold = ds.splits.validation, ds.splits.holdout
    path = tuned_path(cfg, ds.name)
    if not path.exists():
        print(f"No tuned Elo for {ds.name}; run `sfi tune --dataset {ds.name}` first.", file=sys.stderr)
        return 1
    tuned = json.loads(path.read_text(encoding="utf-8"))
    if tuned["snapshot_id"] != ds.snapshot_id or tuned["validation_seasons"] != sorted(val):
        print("Tuned Elo is stale (different data snapshot or validation seasons); re-run `sfi tune`.",
              file=sys.stderr)
        return 1
    tuned_params, default = EloParams(**tuned["best"]["params"]), default_params(cfg)
    c, f = cfg["calibration"], cfg["forecast"]
    gate = GateConfig(f["publish_gate"].get("min_prior_games_per_team") or 0)
    lead = timedelta(hours=f["as_of_lead_hours"])
    last = max(hold) if args.reveal_holdout else max(val)
    games = ds.games_through(last)  # sealed holdout: its games are not even loaded

    factories = [
        CoinFlip, HomeRate,
        lambda: EloModel(default, name="elo_default"),
        lambda: EloModel(tuned_params, name="elo_tuned"),
        lambda: PlattCalibrated(EloModel(tuned_params, name="elo_tuned"), min_samples=c["min_samples"],
                                exclude_first_season=c["exclude_first_season"], name="elo_tuned_platt"),
    ]
    records = run_walkforward(games, factories, source_snapshot_id=ds.snapshot_id,
                              data_kind=ds.data_kind, gate=gate, as_of_lead=lead)
    archive = ForecastArchive(PROJECT_ROOT / cfg["storage"]["backtest_archive"])
    written, same = archive.append(records)
    print(f"Archive: {written} new records, {same} identical re-runs ({len(archive)} total)")

    if args.reveal_holdout:
        log = PROJECT_ROOT / cfg["storage"]["holdout_reveal_log"]
        log.parent.mkdir(parents=True, exist_ok=True)
        entry = {"revealed_at_utc": _now(), "dataset": ds.name, "snapshot_id": ds.snapshot_id,
                 "holdout_seasons": sorted(hold),
                 "models": sorted({f"{r.model_name}@{r.model_version}" for r in records})}
        with open(log, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(entry) + "\n")
        print(f"Holdout revealed; logged to {log.relative_to(PROJECT_ROOT)}")

    v = cfg["validation"]
    by_id = {g.game_id: g for g in games}
    bench = list(ds.benchmarks)
    comparisons = [("elo_tuned", "elo_default"), ("elo_tuned_platt", "elo_tuned"), ("elo_tuned", "home_rate")]
    comparisons += [(m, b) for b in bench for m in ("elo_tuned", "elo_tuned_platt")]
    split_defs = [("validation", val, f"Validation seasons {min(val)}–{max(val)}",
                   "Elo parameters were **selected on these seasons**, so `elo_tuned` scores here are optimistic.")]
    if args.reveal_holdout:
        split_defs.append(("holdout", hold, f"Holdout seasons {min(hold)}–{max(hold)}",
                           "Never used for tuning or any design choice. This is the honest estimate."))
    results, sections = {}, []
    for key, seasons, title, note in split_defs:
        sets = score(records, by_id, set(seasons), benchmarks=ds.benchmarks)
        summary = summarize(sets, v["bootstrap_samples"])
        results[key] = summary
        sections.append(section_lines(
            title, note, sets, summary, comparisons=comparisons,
            reliability=["elo_tuned", "elo_tuned_platt"], n_boot=v["bootstrap_samples"], n_bins=v["calibration_bins"],
        ))
    if not args.reveal_holdout:
        sections.append([f"## Holdout seasons {min(hold)}–{max(hold)}", "",
                         "**Sealed.** Not loaded or evaluated. Reveal only once the model design is frozen "
                         "(`sfi backtest --reveal-holdout`; every reveal is logged).", ""])

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    stem = f"{ds.name}_backtest"
    (reports / f"{stem}.json").write_text(json.dumps(
        {"dataset": ds.name, "data_kind": ds.data_kind, "snapshot_id": ds.snapshot_id,
         "splits": {"burn_in": sorted(ds.splits.burn_in), "validation": sorted(val), "holdout": sorted(hold)},
         "holdout_revealed": args.reveal_holdout, "elo_default": asdict(default),
         "elo_tuned": asdict(tuned_params), "calibration": c, "results": results}, indent=2), encoding="utf-8")
    kind = "synthetic" if ds.data_kind == "synthetic" else "real NFL"
    write_markdown(
        reports / f"{stem}.md", title=f"Walk-forward evaluation: {ds.name} ({kind})",
        data_kind=ds.data_kind, benchmarks=bench,
        header=[f"Data snapshot: `{ds.snapshot_id}`", *ds.notes,
                f"Burn-in (not scored): {sorted(ds.splits.burn_in)}; validation: {min(val)}–{max(val)}; "
                f"holdout: {min(hold)}–{max(hold)} ({'revealed' if args.reveal_holdout else 'sealed'})",
                f"`elo_default`: {asdict(default)}",
                f"`elo_tuned`: {asdict(tuned_params)} (best of {tuned['n_trials']} by validation {tuned['objective']})",
                f"`elo_tuned_platt`: Platt scaling refit before each slate on past issued forecasts only "
                f"(min {c['min_samples']} samples; first season excluded: {c['exclude_first_season']})",
                "Outcome: P(home win); ties excluded from scoring and counted separately. "
                "Includes regular season and playoffs.",
                f"Each kickoff slate forecast {f['as_of_lead_hours']}h before kickoff using only results known by then."],
        sections=sections,
    )
    for key, summary in results.items():
        print(f"[{key}]")
        for name, s in summary.items():
            if s.get("n_scored"):
                print(f"  {name:18s} n={s['n_scored']:5d}  Brier={s['brier']:.4f}  LogLoss={s['log_loss']:.4f}")
    if not args.reveal_holdout:
        print(f"[holdout {min(hold)}-{max(hold)}] sealed")
    print(f"Report: reports/{stem}.md  ({kind})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sfi", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to YAML config (default: config/config.yaml or example)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("synth", help="generate fictional NFL-like seasons")
    sub.add_parser("fetch", help="download a new nflverse schedules snapshot")
    for name in ("tune", "backtest"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", help="dataset name from config (default: config `dataset`)")
        if name == "backtest":
            p.add_argument("--reveal-holdout", action="store_true",
                           help="also evaluate the sealed holdout seasons (logged)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if getattr(args, "dataset", None) is None and args.cmd in ("tune", "backtest"):
        args.dataset = cfg["dataset"]
    handlers = {"synth": cmd_synth, "fetch": cmd_fetch, "tune": cmd_tune, "backtest": cmd_backtest}
    try:
        return handlers[args.cmd](cfg, args)
    except DatasetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
