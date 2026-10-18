"""Command line.

  sfi synth                       generate fictional NFL-like seasons
  sfi fetch                       download a new nflverse schedules snapshot (network)
  sfi tune     [--dataset NAME]   grid-search Elo on validation seasons
  sfi backtest [--dataset NAME] [--reveal-holdout]
                                  walk-forward forecasts for all models; holdout stays sealed unless
                                  --reveal-holdout is given (each reveal is logged)
  sfi forecast [--dry-run]        forecast upcoming real games and append them to the ledger
  sfi score                       score ledger forecasts once results are in a newer snapshot
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from sfi import live
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


def load_tuned(cfg: dict, ds) -> tuple[EloParams, dict] | None:
    """Tuned params for this dataset, or None (with a message) if missing or stale."""
    path = tuned_path(cfg, ds.name)
    if not path.exists():
        print(f"No tuned Elo for {ds.name}; run `sfi tune --dataset {ds.name}` first.", file=sys.stderr)
        return None
    tuned = json.loads(path.read_text(encoding="utf-8"))
    if tuned["validation_seasons"] != sorted(ds.splits.validation):
        print("Validation seasons changed since tuning; re-run `sfi tune`.", file=sys.stderr)
        return None
    return EloParams(**tuned["best"]["params"]), tuned


def elo_factories(cfg: dict, tuned_params: EloParams) -> list:
    c = cfg["calibration"]
    return [
        lambda: EloModel(tuned_params, name="elo_tuned"),
        lambda: PlattCalibrated(EloModel(tuned_params, name="elo_tuned"), min_samples=c["min_samples"],
                                exclude_first_season=c["exclude_first_season"], name="elo_tuned_platt"),
    ]


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
    from sfi.data.sources import nflverse_pbp
    from sfi.datasets import parse_seasons
    nv = cfg["datasets"]["nflverse"]
    for spec, seasons_arg, dir_key in ((nflverse_pbp.PBP, args.pbp, "pbp_dir"),
                                       (nflverse_pbp.INJURIES, args.injuries, "injuries_dir"),
                                       (nflverse_pbp.DEPTH_CHARTS, args.depth_charts, "depth_charts_dir")):
        if not seasons_arg:
            continue
        results = nflverse_pbp.fetch_release(spec, PROJECT_ROOT / nv[dir_key], sorted(parse_seasons(seasons_arg)))
        got = [s for s, _, created in results if created]
        print(f"{spec.tag}: downloaded {len(got)} season(s) {got if got else ''}; {len(results) - len(got)} unchanged.")
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
    loaded = load_tuned(cfg, ds)
    if loaded is None:
        return 1
    tuned_params, tuned = loaded
    if tuned["snapshot_id"] != ds.snapshot_id:
        print("Tuned Elo was fit on a different data snapshot; re-run `sfi tune` before backtesting.",
              file=sys.stderr)
        return 1
    default = default_params(cfg)
    c, f = cfg["calibration"], cfg["forecast"]
    gate = GateConfig(f["publish_gate"].get("min_prior_games_per_team") or 0)
    lead = timedelta(hours=f["as_of_lead_hours"])
    last = max(hold) if args.reveal_holdout else max(val)
    games = ds.games_through(last)  # sealed holdout: its games are not even loaded

    factories = [CoinFlip, HomeRate, lambda: EloModel(default, name="elo_default"),
                 *elo_factories(cfg, tuned_params)]
    feature_note = None
    if args.features == "pbp":
        if ds.name != "nflverse":
            print("--features pbp requires --dataset nflverse", file=sys.stderr)
            return 1
        from sfi.data.sources import nflverse_pbp
        from sfi.features.pbp import load_team_games
        from sfi.models.pbp_logit import PbpLogit
        seasons = sorted({g.season for g in games})  # never beyond the last evaluated season
        files = nflverse_pbp.latest_files(
            PROJECT_ROOT / cfg["datasets"]["nflverse"]["pbp_dir"], seasons)
        pbp_id = nflverse_pbp.combined_snapshot_id([m for _, m in files.values()])
        stats = load_team_games(files, PROJECT_ROOT / "data" / "derived" / "pbp")
        pf = cfg["features"]["pbp"]
        factories += [
            lambda: PbpLogit(tuned_params, stats, metrics=["epa_play"], l2=pf["l2"],
                             min_samples=pf["min_samples"], name="pbp_logit_epa", data_tag=pbp_id),
            lambda: PbpLogit(tuned_params, stats, l2=pf["l2"], min_samples=pf["min_samples"],
                             name="pbp_logit_full", data_tag=pbp_id),
        ]
        feature_note = (f"Play-by-play features: seasons {seasons[0]}–{seasons[-1]}, snapshot `{pbp_id}`, "
                        f"L2 {pf['l2']} (untuned), refit each season start, min {pf['min_samples']} samples. "
                        f"{nflverse_pbp.EPA_CAVEAT}")
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
    if feature_note:
        comparisons += [("pbp_logit_epa", "elo_tuned"), ("pbp_logit_full", "elo_tuned"),
                        ("pbp_logit_full", "pbp_logit_epa")]
        comparisons += [(m, b) for b in bench for m in ("pbp_logit_epa", "pbp_logit_full")]
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
            reliability=["elo_tuned", "pbp_logit_full"] if feature_note else ["elo_tuned", "elo_tuned_platt"],
            n_boot=v["bootstrap_samples"], n_bins=v["calibration_bins"],
        ))
    if not args.reveal_holdout:
        sections.append([f"## Holdout seasons {min(hold)}–{max(hold)}", "",
                         "**Sealed.** Not loaded or evaluated. Reveal only once the model design is frozen "
                         "(`sfi backtest --reveal-holdout`; every reveal is logged).", ""])

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    stem = f"{ds.name}_pbp_backtest" if feature_note else f"{ds.name}_backtest"
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
                f"Each kickoff slate forecast {f['as_of_lead_hours']}h before kickoff using only results known by then.",
                *([feature_note, "`pbp_logit_epa`: Elo logit + offense/defense EPA per play (4-game rolling and "
                   "season-to-date, home minus away). `pbp_logit_full`: Elo logit + all 8 metrics × offense/defense "
                   "× 2 windows (32 differences)."] if feature_note else [])],
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


def _live_dataset(cfg: dict):
    ds = load_dataset(cfg, cfg["live"]["dataset"])
    if ds.data_kind != "real":
        raise DatasetError("prospective forecasting requires a real dataset")
    return ds


def cmd_forecast(cfg: dict, args) -> int:
    ds = _live_dataset(cfg)
    loaded = load_tuned(cfg, ds)
    if loaded is None:
        return 1
    tuned_params, tuned = loaded  # frozen design; may come from an older snapshot by intent
    lv, f = cfg["live"], cfg["forecast"]
    dcfg = cfg["datasets"][ds.name]
    live_cfg = live.LiveConfig(lv["horizon_days"], lv["min_lead_minutes"], lv["max_snapshot_age_hours"],
                               dcfg["known_at_hours"])
    now = datetime.now(timezone.utc).replace(microsecond=0)
    commit, dirty = live.git_state(PROJECT_ROOT)
    manifest = ds.manifest
    try:
        records, run = live.build_run(
            games=ds.games, factories=[HomeRate, *elo_factories(cfg, tuned_params)], now=now, cfg=live_cfg,
            as_of_lead=timedelta(hours=f["as_of_lead_hours"]),
            gate=GateConfig(f["publish_gate"].get("min_prior_games_per_team") or 0),
            snapshot_id=ds.snapshot_id, manifest=manifest, market=ds.benchmarks.get("market_moneyline", {}),
            extra={"dataset": ds.name, "code_commit": commit, "code_dirty": dirty,
                   "tuned_artifact_sha256": file_sha256(tuned_path(cfg, ds.name)),
                   "tuned_params": tuned["best"]["params"], "primary_model": lv["primary_model"],
                   "dry_run": args.dry_run},
        )
    except live.StaleDataError as e:
        print(f"Refusing to forecast on stale data: {e}\nRun `sfi fetch` first.", file=sys.stderr)
        return 1
    if not records:
        print(f"No unplayed games kick off in the next {lv['horizon_days']} days "
              f"(at least {lv['min_lead_minutes']} min from now).")
        return 0

    by_game: dict[str, dict[str, float]] = {}
    for r in records:
        by_game.setdefault(r.game_id, {})[r.model_name] = r.distribution["p_home_win"]
    games = {g.game_id: g for g in ds.games}
    market = run["market_p_home_at_issue"]
    eastern = nflverse.EASTERN
    lines = [f"Issued {run['issued_at_utc']} from snapshot {ds.snapshot_id} "
             f"(retrieved {manifest['retrieved_at_utc']}); code {commit[:7] if commit else '?'}"
             f"{' (uncommitted changes)' if dirty else ''}", "",
             f"| Kickoff (ET) | Game | P(home) {lv['primary_model']} | P(home) Platt | P(home) home_rate | Market (benchmark) |",
             "|---|---|---:|---:|---:|---:|"]
    for gid in run["game_ids"]:
        g, p = games[gid], by_game[gid]
        mk = f"{market[gid]:.3f}" if gid in market else "–"
        site = " (neutral)" if g.neutral_site else ""
        lines.append(f"| {g.scheduled_start_utc.astimezone(eastern):%a %b %d %H:%M} | {g.away_team} @ {g.home_team}{site} | "
                     f"{p.get(lv['primary_model'], float('nan')):.3f} | {p.get('elo_tuned_platt', float('nan')):.3f} | "
                     f"{p.get('home_rate', float('nan')):.3f} | {mk} |")
    print("\n".join(lines))
    if args.dry_run:
        print("\nDry run: nothing archived.")
        return 0
    ledger = live.read_jsonl(PROJECT_ROOT / cfg["storage"]["prospective_ledger"])
    if any(r["forecast_id"] in {x.forecast_id for x in records} for r in ledger):
        print("These forecast IDs are already in the ledger; nothing appended.", file=sys.stderr)
        return 1
    live.append_jsonl(PROJECT_ROOT / cfg["storage"]["prospective_ledger"], [json.loads(r.to_json()) for r in records])
    live.append_jsonl(PROJECT_ROOT / cfg["storage"]["forecast_run_log"], [run])
    report = PROJECT_ROOT / "reports" / "upcoming_forecasts.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text("# Upcoming forecasts (prospective, real NFL)\n\n"
                      "> Issued before kickoff and archived in the ledger. Probabilities are model estimates, "
                      "not guarantees. Market column is a benchmark only, never a model input.\n\n"
                      + "\n".join(lines) + f"\n\n{nflverse.ATTRIBUTION}\n", encoding="utf-8")
    print(f"\nArchived {len(records)} forecasts for {len(run['game_ids'])} games (run {run['run_id']}) "
          f"to {cfg['storage']['prospective_ledger']}. Commit the ledger to keep an external timestamp.")
    return 0


def cmd_score(cfg: dict, args) -> int:
    ds = _live_dataset(cfg)
    ledger = live.read_jsonl(PROJECT_ROOT / cfg["storage"]["prospective_ledger"])
    if not ledger:
        print("No prospective forecasts in the ledger yet; run `sfi forecast`.")
        return 0
    runs = live.read_jsonl(PROJECT_ROOT / cfg["storage"]["forecast_run_log"])
    games = {g.game_id: g for g in ds.games}
    bench = {"market_at_issue": live.market_at_issue(runs, ledger, games),
             "market_moneyline": ds.benchmarks.get("market_moneyline", {})}
    res = live.score_ledger(ledger, games, bench)
    v = cfg["validation"]
    summary = summarize(res.sets, v["bootstrap_samples"])
    n = max((s.get("n_scored", 0) for s in summary.values()), default=0)
    note = (f"Latest forecast issued before actual kickoff, per model and game. {res.pending} game(s) still "
            f"awaiting results. Excluded records: {res.excluded}.")
    if n < 100:
        note += f" **Only {n} scored games: far too few to judge skill or calibration.**"
    comparisons = [("elo_tuned", "home_rate"), ("elo_tuned_platt", "elo_tuned"),
                   ("elo_tuned", "market_at_issue"), ("elo_tuned", "market_moneyline")]
    section = section_lines("Prospective forecasts", note, res.sets, summary, comparisons=comparisons,
                            reliability=["elo_tuned"] if n >= 100 else [],
                            n_boot=v["bootstrap_samples"], n_bins=v["calibration_bins"])
    out = PROJECT_ROOT / "reports" / "prospective_scores.md"
    write_markdown(out, title="Prospective scoring (real NFL, issued before kickoff)", data_kind="real",
                   benchmarks=["market_moneyline"],
                   header=[f"Scored against snapshot `{ds.snapshot_id}` ({ds.notes[0]})",
                           f"Ledger: {len(ledger)} forecast records from {len(runs)} run(s)",
                           "`market_at_issue`: market line recorded when our forecast was issued; "
                           "`market_moneyline`: line in the scoring snapshot (likely closing)."],
                   sections=[section])
    for name, s in summary.items():
        if s.get("n_scored"):
            print(f"  {name:18s} n={s['n_scored']:4d}  Brier={s['brier']:.4f}  LogLoss={s['log_loss']:.4f}")
    print(f"{res.pending} game(s) pending; excluded {res.excluded}")
    print("Report: reports/prospective_scores.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sfi", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to YAML config (default: config/config.yaml or example)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("synth", help="generate fictional NFL-like seasons")
    p = sub.add_parser("fetch", help="download a new nflverse schedules snapshot")
    p.add_argument("--pbp", metavar="SEASONS", help='also fetch play-by-play, e.g. "2000-2026" or "2026"')
    p.add_argument("--injuries", metavar="SEASONS", help='also fetch injury reports (2009+)')
    p.add_argument("--depth-charts", metavar="SEASONS", help='also fetch depth charts (2001+)')
    p = sub.add_parser("forecast", help="forecast upcoming games and archive them (real data)")
    p.add_argument("--dry-run", action="store_true", help="print forecasts without archiving")
    sub.add_parser("score", help="score archived prospective forecasts against the latest snapshot")
    for name in ("tune", "backtest"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", help="dataset name from config (default: config `dataset`)")
        if name == "backtest":
            p.add_argument("--reveal-holdout", action="store_true",
                           help="also evaluate the sealed holdout seasons (logged)")
            p.add_argument("--features", choices=["pbp"], help="also evaluate play-by-play feature models")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if getattr(args, "dataset", None) is None and args.cmd in ("tune", "backtest"):
        args.dataset = cfg["dataset"]
    handlers = {"synth": cmd_synth, "fetch": cmd_fetch, "tune": cmd_tune, "backtest": cmd_backtest,
                "forecast": cmd_forecast, "score": cmd_score}
    try:
        return handlers[args.cmd](cfg, args)
    except DatasetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
