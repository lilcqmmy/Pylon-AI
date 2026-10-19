"""Command line.

  sfi synth                       generate fictional NFL-like seasons
  sfi fetch                       download a new nflverse schedules snapshot (network)
  sfi tune     [--dataset NAME]   grid-search Elo on validation seasons
  sfi backtest [--dataset NAME] [--reveal-holdout]
                                  walk-forward forecasts for all models; holdout stays sealed unless
                                  --reveal-holdout is given (each reveal is logged)
  sfi forecast [--dry-run]        forecast upcoming real games and append them to the ledger
  sfi score                       score ledger forecasts once results are in a newer snapshot
  sfi backtest-scores [--dataset NAME] [--reveal-holdout]
                                  walk-forward score-margin and game-total distributions
  sfi backtest-players [--reveal-holdout]
                                  walk-forward passing/rushing/receiving yard distributions
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


def challenger_factories(cfg: dict, tuned_params: EloParams) -> list:
    """Pre-registered challengers (docs/challenger_elo_hfa.md)."""
    from sfi.models.elo import EloAdaptiveHome
    k = cfg["live"]["elo_hfa_home_k"]
    return [lambda: EloAdaptiveHome(tuned_params, home_k=k, name="elo_hfa")]


def elo_factories(cfg: dict, tuned_params: EloParams) -> list:
    c = cfg["calibration"]
    return [
        lambda: EloModel(tuned_params, name="elo_tuned"),
        lambda: PlattCalibrated(EloModel(tuned_params, name="elo_tuned"), min_samples=c["min_samples"],
                                exclude_first_season=c["exclude_first_season"], name="elo_tuned_platt"),
    ]


def _holdout_note(cfg: dict, ds, hold, kind: str) -> list[str]:
    """Report section for a run that does not evaluate the holdout: sealed, or already revealed earlier."""
    log = PROJECT_ROOT / cfg["storage"]["holdout_reveal_log"]
    prior = []
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("dataset") == ds.name and d.get("kind", "win") == kind:
                    prior.append(d["revealed_at_utc"])
    title = f"## Holdout {min(hold)}–{max(hold)}"
    if prior:
        return [title, "", f"**Already revealed** ({', '.join(prior)}); not evaluated in this run. Results: "
                "`docs/holdout_results.md` and the dated `*_holdout_*.md` report copies. The holdout can no "
                "longer serve as an unbiased test; judge new designs on prospective forecasts.", ""]
    return [title, "", "**Sealed.** Not loaded or evaluated. Reveal only once the model design is frozen "
            "(`--reveal-holdout`; every reveal is logged).", ""]


def _save_holdout_copy(path: Path) -> None:
    """Keep a dated copy of a report that includes the holdout, so later runs don't overwrite it."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    copy = path.with_name(f"{path.stem}_holdout_{stamp}{path.suffix}")
    copy.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


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
    if args.live:  # current season's play-by-play and injuries, for live player forecasts
        import csv as _csv
        with open(folder / "games.csv", newline="", encoding="utf-8") as fh:
            current = str(max(int(r["season"]) for r in _csv.DictReader(fh)))
        args.pbp = args.pbp or current
        args.injuries = args.injuries or current
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
                 *elo_factories(cfg, tuned_params), *challenger_factories(cfg, tuned_params)]
    feats = args.features or []
    feature_notes: list[str] = []
    if feats and ds.name != "nflverse":
        print("--features requires --dataset nflverse", file=sys.stderr)
        return 1
    if "pbp" in feats:
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
        feature_notes.append(f"Play-by-play features: seasons {seasons[0]}–{seasons[-1]}, snapshot `{pbp_id}`, "
                        f"L2 {pf['l2']} (untuned), refit each season start, min {pf['min_samples']} samples. "
                        f"{nflverse_pbp.EPA_CAVEAT}")
        feature_notes.append("`pbp_logit_epa`: Elo logit + offense/defense EPA per play (4-game rolling and "
                             "season-to-date, home minus away). `pbp_logit_full`: Elo logit + all 8 metrics × "
                             "offense/defense × 2 windows (32 differences).")
    if "qb" in feats:
        from sfi.data.sources import nflverse, nflverse_pbp
        from sfi.features.qb import load_injuries, load_qb_games, load_starters
        from sfi.models.qb_adjust import QbAdjusted
        nv, qf = cfg["datasets"]["nflverse"], cfg["features"]["qb"]
        seasons = sorted({g.season for g in games})  # never beyond the last evaluated season
        pbp_files = nflverse_pbp.latest_files(PROJECT_ROOT / nv["pbp_dir"], seasons)
        inj_seasons = [s for s in seasons if s >= 2009]
        inj_files = nflverse_pbp.latest_release_files(nflverse_pbp.INJURIES, PROJECT_ROOT / nv["injuries_dir"], inj_seasons)
        qb_games = load_qb_games(pbp_files, PROJECT_ROOT / "data" / "derived" / "pbp")
        starters = load_starters(nflverse.latest_snapshot(PROJECT_ROOT / nv["raw_dir"]) / "games.csv", games)
        inj = load_injuries(inj_files, games, missing_hours=qf["missing_timestamp_hours"])
        qb_tag = (nflverse_pbp.combined_snapshot_id([m for _, m in pbp_files.values()]) + "|"
                  + nflverse_pbp.combined_snapshot_id([m for _, m in inj_files.values()]))
        factories.append(lambda: QbAdjusted(tuned_params, qb_games, starters, inj.rows, k_prior=qf["k_prior"],
                                            prior_mean=qf["prior_mean"], l2=qf["l2"], min_samples=qf["min_samples"],
                                            name="elo_qb", data_tag=qb_tag))
        feature_notes.append(
            f"`elo_qb`: Elo logit + QB availability (expected vs previous starter rating, Out/Doubtful and "
            f"Questionable flags), ridge L2 {qf['l2']} refit each season; QB rating = prior qb_epa/dropback shrunk "
            f"to {qf['prior_mean']} with {qf['k_prior']:g} dropbacks (untuned). Injury rows (QBs, {inj_seasons[0]}–"
            f"{inj_seasons[-1]}): {sum(len(v) for v in inj.rows.values())} used, {inj.policy_rows} with policy "
            f"timestamp (kickoff − {qf['missing_timestamp_hours']:g}h), {inj.dropped_after_kickoff} dropped "
            f"(modified after kickoff), {inj.unmatched} unmatched. Injury data starts 2009, so compare on 2010–2019.")
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
    comparisons = [("elo_tuned", "elo_default"), ("elo_tuned_platt", "elo_tuned"), ("elo_tuned", "home_rate"),
                   ("elo_hfa", "elo_tuned")]
    comparisons += [(m, b) for b in bench for m in ("elo_tuned", "elo_tuned_platt")]
    if "pbp" in feats:
        comparisons += [("pbp_logit_epa", "elo_tuned"), ("pbp_logit_full", "elo_tuned"),
                        ("pbp_logit_full", "pbp_logit_epa")]
        comparisons += [(m, b) for b in bench for m in ("pbp_logit_epa", "pbp_logit_full")]
    if "qb" in feats:
        comparisons += [("elo_qb", "elo_tuned")] + [("elo_qb", b) for b in bench]
    split_defs = [("validation", val, f"Validation seasons {min(val)}–{max(val)}",
                   "Elo parameters were **selected on these seasons**, so `elo_tuned` scores here are optimistic.")]
    if "qb" in feats:
        inj_val = {s for s in val if s >= 2010}
        split_defs.append(("validation_2010_2019", inj_val, f"Validation seasons {min(inj_val)}–{max(inj_val)} "
                           "(seasons with timestamped injury reports)",
                           "The comparison for `elo_qb`: injury-report timestamps exist from 2010. "
                           "Same Elo-selection caveat as above."))
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
            reliability=(["elo_tuned"] + [m for f, m in (("pbp", "pbp_logit_full"), ("qb", "elo_qb")) if f in feats]
                         if feats else ["elo_tuned", "elo_tuned_platt"]),
            n_boot=v["bootstrap_samples"], n_bins=v["calibration_bins"],
        ))
    if not args.reveal_holdout:
        sections.append(_holdout_note(cfg, ds, hold, "win"))

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    stem = f"{ds.name}_{'_'.join(sorted(feats))}_backtest" if feats else f"{ds.name}_backtest"
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
                *feature_notes],
        sections=sections,
    )
    if args.reveal_holdout:
        _save_holdout_copy(reports / f"{stem}.md")
    for key, summary in results.items():
        print(f"[{key}]")
        for name, s in summary.items():
            if s.get("n_scored"):
                print(f"  {name:18s} n={s['n_scored']:5d}  Brier={s['brier']:.4f}  LogLoss={s['log_loss']:.4f}")
    if not args.reveal_holdout:
        print(f"[holdout {min(hold)}-{max(hold)}] sealed")
    print(f"Report: reports/{stem}.md  ({kind})")
    return 0


def cmd_backtest_scores(cfg: dict, args) -> int:
    from sfi.evaluation import scores_eval as SE
    from sfi.models.scores import EloScores, NaiveScores, PointsModel, PointsParams

    ds = load_dataset(cfg, args.dataset)
    val, hold = ds.splits.validation, ds.splits.holdout
    loaded = load_tuned(cfg, ds)
    if loaded is None:
        return 1
    tuned_params, tuned = loaded
    if tuned["snapshot_id"] != ds.snapshot_id:
        print("Tuned Elo was fit on a different data snapshot; re-run `sfi tune` before backtesting.", file=sys.stderr)
        return 1
    sc = cfg["scores"]
    pp = PointsParams(**sc["points"])
    lead = timedelta(hours=cfg["forecast"]["as_of_lead_hours"])
    last = max(hold) if args.reveal_holdout else max(val)
    games = ds.games_through(last)
    factories = [
        lambda: NaiveScores(),
        lambda: EloScores(tuned_params, residual_method="normal", window=sc["residual_window"], min_samples=sc["min_samples"]),
        lambda: EloScores(tuned_params, residual_method="empirical", window=sc["residual_window"], min_samples=sc["min_samples"]),
        lambda: PointsModel(pp, residual_method="normal", window=sc["residual_window"], min_samples=sc["min_samples"]),
        lambda: PointsModel(pp, residual_method="empirical", window=sc["residual_window"], min_samples=sc["min_samples"]),
    ]
    forecasts, records = [], []
    for factory in factories:
        model = factory()
        for fc in SE.iter_score_walkforward(games, model, as_of_lead=lead):
            forecasts.append((model.name, fc))
            records.append(SE.make_score_record(fc, model, source_snapshot_id=ds.snapshot_id, data_kind=ds.data_kind))
    archive = ForecastArchive(PROJECT_ROOT / cfg["storage"]["score_archive"])
    written, same = archive.append(records)
    print(f"Archive: {written} new records, {same} identical re-runs ({len(archive)} total)")
    if args.reveal_holdout:
        log = PROJECT_ROOT / cfg["storage"]["holdout_reveal_log"]
        with open(log, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"revealed_at_utc": _now(), "dataset": ds.name, "snapshot_id": ds.snapshot_id,
                                 "holdout_seasons": sorted(hold), "kind": "scores",
                                 "models": sorted({f"{r.model_name}@{r.model_version}" for r in records})}) + "\n")
        print(f"Holdout revealed; logged to {log.relative_to(PROJECT_ROOT)}")

    n_boot = cfg["validation"]["bootstrap_samples"]
    by_id = {g.game_id: g for g in games}
    splits = [("validation", set(val), f"Validation {min(val)}–{max(val)}",
               "Elo parameters were selected on these seasons; points-model and spread settings are untuned defaults.")]
    if args.reveal_holdout:
        splits.append(("holdout", set(hold), f"Holdout {min(hold)}–{max(hold)}", "Never used for any design choice."))
    sections, results = [], {}
    for key, seasons, title, note in splits:
        sections.append([f"## {title}", ""])
        for target in ("margin", "total"):
            evals = SE.evaluate(forecasts, target, seasons)
            summary = SE.summarize_target(evals, n_boot)
            market = SE.market_point_eval(target, ds.lines, by_id, seasons) if ds.lines else None
            mname = market.model if market is not None else None
            best = ["scores_points_empirical", "scores_points_normal"] if target == "total" else \
                   ["scores_elo_normal", "scores_points_empirical"]
            comparisons = [(m, "scores_naive") for m in best] + [
                ("scores_points_empirical", "scores_points_normal"), ("scores_elo_empirical", "scores_elo_normal")]
            if target == "margin":
                comparisons.append(("scores_points_empirical", "scores_elo_normal"))
            if mname:
                comparisons += [(m, mname) for m in best]
            results[f"{key}_{target}"] = summary
            sections.append(SE.section_lines(target, title, "" if target == "total" else note, evals, summary,
                                             market, comparisons, n_boot))
            # implied win probability from the margin distribution, for consistency with the win models
            if target == "margin":
                rows = ["Implied P(home win) from the margin distribution (Brier, non-tied games):", ""]
                for name in summary:
                    ps, ys = [], []
                    for nm, fc in forecasts:
                        g = fc.game
                        if nm == name and g.season in seasons and g.is_final and g.home_outcome != 0.5:
                            ps.append(SE.implied_p_home(fc.margin))
                            ys.append(g.home_outcome)
                    from sfi.evaluation.metrics import brier
                    rows.append(f"- `{name}`: {brier(ps, ys):.4f} (n = {len(ys)})")
                sections.append(rows + [""])
    if not args.reveal_holdout:
        sections.append(_holdout_note(cfg, ds, hold, "scores"))

    stem = f"{ds.name}_scores_backtest"
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    (PROJECT_ROOT / "reports" / f"{stem}.json").write_text(json.dumps(
        {"dataset": ds.name, "snapshot_id": ds.snapshot_id, "holdout_revealed": args.reveal_holdout,
         "points_params": asdict(pp), "results": results}, indent=2), encoding="utf-8")
    write_markdown(
        PROJECT_ROOT / "reports" / f"{stem}.md", title=f"Margin and total distributions: {ds.name}",
        data_kind=ds.data_kind, benchmarks=[],
        header=[f"Data snapshot: `{ds.snapshot_id}`", *ds.notes,
                f"Each kickoff slate forecast {cfg['forecast']['as_of_lead_hours']}h before kickoff using only results known by then.",
                "`scores_naive`: past margins (last 2,000 games) and totals (last 512 games), no game information.",
                f"`scores_elo_*`: margin mean = OLS on tuned-Elo points edge, refit each season; total mean = last 512 games.",
                f"`scores_points_*`: team offense/defense scoring ratings {asdict(pp)} (untuned).",
                f"`*_normal`: discretized Normal with sd of the last {sc['residual_window']} residuals; `*_empirical`: "
                "shape of those residuals (keeps key-number mass). Spreads use the model's own past forecasts only.",
                "Market spread/total lines are point benchmarks only (recording time undocumented, likely near kickoff)."],
        sections=sections,
    )
    if args.reveal_holdout:
        _save_holdout_copy(PROJECT_ROOT / "reports" / f"{stem}.md")
    for k, summ in results.items():
        print(f"[{k}]")
        for name, s in summ.items():
            cov = " ".join(f"{lv}:{s['coverage'][str(lv)]['observed']:.3f}/{s['coverage'][str(lv)]['claimed']:.3f}"
                           for lv in SE.LEVELS)
            print(f"  {name:24s} n={s['n']:5d} CRPS={s['crps']:.3f} MAE={s['mae_median']:.2f} cov {cov} PITchi2={s['pit_chi2']:.1f}")
    print(f"Report: reports/{stem}.md")
    return 0


def cmd_backtest_players(cfg: dict, args) -> int:
    from sfi.data.sources import nflverse_pbp
    from sfi.evaluation import players_eval as PE
    from sfi.evaluation import scores_eval as SE
    from sfi.features.players import STATS, load_player_games
    from sfi.features.qb import load_injuries
    from sfi.models.players import PlayerOppEff, PlayerParams, PlayerRecentAvg

    ds = load_dataset(cfg, args.dataset)
    if ds.name != "nflverse":
        print("backtest-players requires --dataset nflverse", file=sys.stderr)
        return 1
    val, hold = ds.splits.validation, ds.splits.holdout
    nv, pc = cfg["datasets"]["nflverse"], cfg["players"]
    last = max(hold) if args.reveal_holdout else max(val)
    games = ds.games_through(last)
    seasons = sorted({g.season for g in games})
    pbp_files = nflverse_pbp.latest_files(PROJECT_ROOT / nv["pbp_dir"], seasons)
    inj_files = nflverse_pbp.latest_release_files(nflverse_pbp.INJURIES, PROJECT_ROOT / nv["injuries_dir"],
                                                  [s for s in seasons if s >= 2009])
    stats = load_player_games(pbp_files, PROJECT_ROOT / "data" / "derived" / "pbp")
    inj = load_injuries(inj_files, games, missing_hours=cfg["features"]["qb"]["missing_timestamp_hours"], positions=None)
    tag = (nflverse_pbp.combined_snapshot_id([m for _, m in pbp_files.values()]) + "|"
           + nflverse_pbp.combined_snapshot_id([m for _, m in inj_files.values()]))
    params = PlayerParams(**pc["params"])
    lead = timedelta(hours=cfg["forecast"]["as_of_lead_hours"])
    forecasts, records = [], []
    for factory in (lambda: PlayerRecentAvg(stats, inj.rows, params, data_tag=tag),
                    lambda: PlayerOppEff(stats, inj.rows, params, data_tag=tag)):
        model = factory()
        for as_of, fc in PE.iter_player_walkforward(games, model, as_of_lead=lead):
            forecasts.append((model.name, as_of, fc))
            records.append(PE.make_player_record(as_of, fc, model, source_snapshot_id=ds.snapshot_id, data_kind=ds.data_kind))
    archive = ForecastArchive(PROJECT_ROOT / cfg["storage"]["player_archive"])
    written, same = archive.append(records)
    print(f"Archive: {written} new records, {same} identical re-runs ({len(archive)} total)")
    if args.reveal_holdout:
        log = PROJECT_ROOT / cfg["storage"]["holdout_reveal_log"]
        with open(log, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"revealed_at_utc": _now(), "dataset": ds.name, "snapshot_id": ds.snapshot_id,
                                 "holdout_seasons": sorted(hold), "kind": "players",
                                 "models": sorted({f"{r.model_name}@{r.model_version}" for r in records})}) + "\n")

    n_boot = cfg["validation"]["bootstrap_samples"]
    inj_val = {s for s in val if s >= 2010}
    splits = [("validation_2010_2019", inj_val, f"Validation {min(inj_val)}–{max(inj_val)} (injury reports available)")]
    if args.reveal_holdout:
        splits.append(("holdout", set(hold), f"Holdout {min(hold)}–{max(hold)}"))
    sections, results = [], {}
    stat_label = {"pass": "passing yards", "rush": "rushing yards", "rec": "receiving yards"}
    for key, sset, title in splits:
        evals, extra = PE.evaluate_players(forecasts, stats, sset)
        sections.append([f"## {title}", ""])
        for stat in STATS:
            ev = {name: e for (name, s), e in evals.items() if s == stat}
            if not ev:
                continue
            summ = SE.summarize_target(ev, n_boot)
            zero = {name: extra[(name, s)]["no_opportunity"] for (name, s) in evals if s == stat}
            results[f"{key}_{stat}"] = {"summary": summ, "no_opportunity": zero}
            note = (f"Pregame pool: at least {params.min_opp[stat]} opportunities for the team in its last "
                    f"{params.pool_games} games, still on the team, not listed Out/Doubtful before as-of. "
                    f"Pool players with no opportunity (scored as 0): {zero}.")
            sections.append(SE.section_lines(stat_label[stat], title, note, ev, summ, None,
                                             [("players_opp_eff", "players_recent_avg")], n_boot))
    if not args.reveal_holdout:
        sections.append(_holdout_note(cfg, ds, hold, "players"))
    stem = f"{ds.name}_players_backtest"
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    (PROJECT_ROOT / "reports" / f"{stem}.json").write_text(json.dumps(
        {"dataset": ds.name, "snapshot_id": ds.snapshot_id, "data_tag": tag, "params": asdict(params),
         "results": results}, indent=2, default=str), encoding="utf-8")
    write_markdown(PROJECT_ROOT / "reports" / f"{stem}.md", title=f"Player yardage distributions: {ds.name}",
                   data_kind=ds.data_kind, benchmarks=[],
                   header=[f"Data snapshot: `{ds.snapshot_id}`; play-by-play + injuries `{tag}`", *ds.notes,
                           f"Forecasts per kickoff slate {cfg['forecast']['as_of_lead_hours']}h before kickoff; "
                           "player stats read only from released games.",
                           "`players_recent_avg`: mean of last 4 appearances. `players_opp_eff`: recent volume × shrunk "
                           "career yards/opportunity × opponent allowed-rate adjustment.",
                           f"Spread: model's own past residuals at a similar mean (at least {params.min_k}). Parameters: "
                           f"{asdict(params)} (untuned). No player-prop market data is available for comparison.",
                           "Yardage excludes laterals; matches official 2019 season totals for the yardage leaders."],
                   sections=sections)
    if args.reveal_holdout:
        _save_holdout_copy(PROJECT_ROOT / "reports" / f"{stem}.md")
    for k, r in results.items():
        for name, s in r["summary"].items():
            cov = " ".join(f"{lv}:{s['coverage'][str(lv)]['observed']:.3f}/{s['coverage'][str(lv)]['claimed']:.3f}" for lv in SE.LEVELS)
            print(f"[{k}] {name:20s} n={s['n']:6d} CRPS={s['crps']:.2f} MAE={s['mae_median']:.1f} cov {cov} PIT={s['pit_chi2']:.0f}")
    print(f"Report: reports/{stem}.md")
    return 0


def _live_dataset(cfg: dict):
    ds = load_dataset(cfg, cfg["live"]["dataset"])
    if ds.data_kind != "real":
        raise DatasetError("prospective forecasting requires a real dataset")
    return ds


def _player_inputs(cfg: dict, ds, games):
    """Play-by-play player stats (all seasons in `games`) and all-position injury rows, plus a data tag."""
    from sfi.data.sources import nflverse_pbp
    from sfi.features.players import load_player_games
    from sfi.features.qb import load_injuries
    nv = cfg["datasets"]["nflverse"]
    seasons = sorted({g.season for g in games})
    pbp_files = nflverse_pbp.latest_files(PROJECT_ROOT / nv["pbp_dir"], seasons)
    inj_files = nflverse_pbp.latest_release_files(nflverse_pbp.INJURIES, PROJECT_ROOT / nv["injuries_dir"],
                                                  [s for s in seasons if s >= 2009])
    stats = load_player_games(pbp_files, PROJECT_ROOT / "data" / "derived" / "pbp")
    inj = load_injuries(inj_files, games, missing_hours=cfg["features"]["qb"]["missing_timestamp_hours"], positions=None)
    tag = (nflverse_pbp.combined_snapshot_id([m for _, m in pbp_files.values()]) + "|"
           + nflverse_pbp.combined_snapshot_id([m for _, m in inj_files.values()]))
    return stats, inj, tag


def _fmt_dist(d: dict | None, target: str) -> str:
    if not d:
        return "–"
    s = d[target]
    return f"{s['q50']:+d} [{s['q10']:+d}, {s['q90']:+d}]" if target == "margin" else f"{s['q50']} [{s['q10']}, {s['q90']}]"


def _player_table(records, games: dict, thresholds: dict, status: dict | None, names: dict | None = None) -> list[str]:
    label = {"pass": "passing", "rush": "rushing", "rec": "receiving"}
    rows = [r for r in records if r.outcome_type.startswith("player_")]
    if not rows:
        return [f"No player forecasts ({status})."]
    out = ["| Kickoff (ET) | Game | Team | Player | Stat | Median [80%] | P(≥ x) |", "|---|---|---|---|---|---:|---|"]
    from sfi.data.sources import nflverse as _nv
    order = {"pass": 0, "rush": 1, "rec": 2}
    for r in sorted(rows, key=lambda r: (r.scheduled_start_utc, r.game_id, r.distribution["team"],
                                         order[r.outcome_type.split("_")[1]], -r.distribution["mu"])):
        d, g = r.distribution, games[r.game_id]
        stat = r.outcome_type.split("_")[1]
        p = ", ".join(f"{x}: {v:.2f}" for x, v in d.get("p_at_least", {}).items())
        out.append(f"| {g.scheduled_start_utc.astimezone(_nv.EASTERN):%a %b %d %H:%M} | {g.away_team} @ {g.home_team} | "
                   f"{d['team']} | {(names or {}).get(d['player'], d['player'])} | {label[stat]} | {d['q50']} [{d['q10']}, {d['q90']}] | {p} |")
    return out


def _fmt_lines(ln: dict | None) -> str:
    if not ln:
        return "–"
    sp, tot = ln.get("spread"), ln.get("total")
    return f"{'' if sp is None else f'{sp:+g}'} / {'' if tot is None else f'{tot:g}'}"


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
    from sfi.models.scores import NaiveScores, PointsModel, PointsParams
    sc = cfg["scores"]
    pp = PointsParams(**sc["points"])
    score_factories = [
        lambda: NaiveScores(),
        lambda: PointsModel(pp, residual_method="empirical", window=sc["residual_window"], min_samples=sc["min_samples"]),
    ]
    from sfi.models.players import PlayerOppEff, PlayerParams
    try:
        pstats, pinj, ptag = _player_inputs(cfg, ds, ds.games)
    except (FileNotFoundError, ValueError) as e:
        print(f"Player forecasts unavailable ({e}); run `sfi fetch --live`.", file=sys.stderr)
        pstats = None
    player_factory = None
    if pstats is not None:
        pparams = PlayerParams(**cfg["players"]["params"])
        player_factory = lambda: PlayerOppEff(pstats, pinj.rows, pparams, data_tag=ptag)  # noqa: E731
    try:
        records, run = live.build_run(
            games=ds.games, factories=[HomeRate, *elo_factories(cfg, tuned_params), *challenger_factories(cfg, tuned_params)],
            now=now, cfg=live_cfg,
            as_of_lead=timedelta(hours=f["as_of_lead_hours"]),
            gate=GateConfig(f["publish_gate"].get("min_prior_games_per_team") or 0),
            snapshot_id=ds.snapshot_id, manifest=manifest, market=ds.benchmarks.get("market_moneyline", {}),
            extra={"dataset": ds.name, "code_commit": commit, "code_dirty": dirty,
                   "tuned_artifact_sha256": file_sha256(tuned_path(cfg, ds.name)),
                   "tuned_params": tuned["best"]["params"], "primary_model": lv["primary_model"],
                   "dry_run": args.dry_run, "primary_score_model": lv["primary_score_model"]},
            score_factories=score_factories, lines=ds.lines, player_factory=player_factory,
            player_stats=pstats, thresholds=lv["player_thresholds"], max_pbp_gap_hours=lv["max_pbp_gap_hours"],
        )
    except live.StaleDataError as e:
        print(f"Refusing to forecast on stale data: {e}\nRun `sfi fetch` first.", file=sys.stderr)
        return 1
    if not records:
        print(f"No unplayed games kick off in the next {lv['horizon_days']} days "
              f"(at least {lv['min_lead_minutes']} min from now).")
        return 0

    by_game: dict[str, dict[str, float]] = {}
    dist: dict[str, dict] = {}
    for r in records:
        if r.outcome_type == "home_win_excl_tie":
            by_game.setdefault(r.game_id, {})[r.model_name] = r.distribution["p_home_win"]
        elif r.outcome_type == "home_margin_and_total" and r.model_name == lv["primary_score_model"]:
            dist[r.game_id] = r.distribution
    games = {g.game_id: g for g in ds.games}
    market = run["market_p_home_at_issue"]
    eastern = nflverse.EASTERN
    lines = [f"Issued {run['issued_at_utc']} from snapshot {ds.snapshot_id} "
             f"(retrieved {manifest['retrieved_at_utc']}); code {commit[:7] if commit else '?'}"
             f"{' (uncommitted changes)' if dirty else ''}", "",
             f"| Kickoff (ET) | Game | P(home) {lv['primary_model']} | P(home) Platt | P(home) elo_hfa (challenger) | P(home) home_rate | Market (benchmark) "
             f"| Margin median [80%] | Total median [80%] | Market spread / total |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    mlines = run.get("market_lines_at_issue", {})
    for gid in run["game_ids"]:
        g, p = games[gid], by_game[gid]
        mk = f"{market[gid]:.3f}" if gid in market else "–"
        site = " (neutral)" if g.neutral_site else ""
        lines.append(f"| {g.scheduled_start_utc.astimezone(eastern):%a %b %d %H:%M} | {g.away_team} @ {g.home_team}{site} | "
                     f"{p.get(lv['primary_model'], float('nan')):.3f} | {p.get('elo_tuned_platt', float('nan')):.3f} | "
                     f"{p.get('elo_hfa', float('nan')):.3f} | "
                     f"{p.get('home_rate', float('nan')):.3f} | {mk} | {_fmt_dist(dist.get(gid), 'margin')} | "
                     f"{_fmt_dist(dist.get(gid), 'total')} | {_fmt_lines(mlines.get(gid))} |")
    print("\n".join(lines))
    names = {}
    if pstats is not None:
        from sfi.data.sources import nflverse_pbp
        from sfi.features.players import player_names
        cur = max(g.season for g in ds.games)
        names = player_names([nflverse_pbp.latest_files(PROJECT_ROOT / cfg["datasets"]["nflverse"]["pbp_dir"], [s])[s][0]
                              for s in (cur - 1, cur)])
    player_lines = _player_table(records, games, lv["player_thresholds"], run.get("players"), names)
    n_players = sum(1 for r in records if r.outcome_type.startswith("player_"))
    print(f"\nPlayer yardage forecasts: {n_players} ({run.get('players')})")
    qb_rows = [ln for ln in player_lines if "| passing |" in ln]
    if qb_rows:
        print("\n".join(player_lines[:2] + qb_rows[:12]))
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
                      "not guarantees. Market columns are benchmarks only, never model inputs. Margin is home "
                      "minus away points; 80% = central interval of the predictive distribution.\n\n"
                      + "\n".join(lines) + f"\n\n{nflverse.ATTRIBUTION}\n", encoding="utf-8")
    (PROJECT_ROOT / "reports" / "upcoming_player_forecasts.md").write_text(
        "# Upcoming player yardage forecasts (prospective, real NFL)\n\n"
        "> Issued before kickoff and archived in the ledger. Pregame pool: players with recent usage for the team, "
        "not listed Out/Doubtful when issued; passing = expected starter only. P(≥ x) = model probability of at "
        "least x yards; estimates, not guarantees. No player-prop market data is used.\n\n"
        + "\n".join(player_lines) + f"\n\n{nflverse.ATTRIBUTION}\n", encoding="utf-8")
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
    comparisons = [("elo_hfa", "elo_tuned"), ("elo_tuned", "home_rate"), ("elo_tuned_platt", "elo_tuned"),
                   ("elo_tuned", "market_at_issue"), ("elo_tuned", "market_moneyline"), ("elo_hfa", "market_at_issue")]
    section = section_lines("Prospective forecasts", note, res.sets, summary, comparisons=comparisons,
                            reliability=["elo_tuned"] if n >= 100 else [],
                            n_boot=v["bootstrap_samples"], n_bins=v["calibration_bins"])

    from sfi.evaluation import scores_eval as SE
    dres = live.score_ledger_distributions(ledger, runs, games, ds.lines)
    dist_sections, dist_counts = [["## Margin and total distributions", ""]], {}
    for target in ("margin", "total"):
        evals = dres.evals[target]
        if not evals:
            continue
        dsum = SE.summarize_target(evals, v["bootstrap_samples"])
        nd = max(s["n"] for s in dsum.values())
        dist_counts[target] = {k: s for k, s in dsum.items()}
        dnote = (f"Latest pre-kickoff forecast per model and game, scored exactly from the archived pmf. "
                 f"{dres.pending} game(s) pending; excluded {dres.excluded}.")
        if nd < 100:
            dnote += f" **Only {nd} scored games: far too few to judge accuracy or coverage.**"
        mk = dres.market[target]
        key = "spread" if target == "margin" else "total"
        cmp_ = [("scores_points_empirical", "scores_naive"),
                ("scores_points_empirical", f"market_{key}_at_issue")]
        lines_ = SE.section_lines(target, "Prospective", dnote, {**evals, mk["at_issue"].model: mk["at_issue"]},
                                  dsum, mk["snapshot"], cmp_, v["bootstrap_samples"])
        dist_sections.append(lines_)
    player_sections = []
    try:
        pstats, _, _ = _player_inputs(cfg, ds, ds.games)
    except (FileNotFoundError, ValueError) as e:
        pstats = None
        player_sections.append(["## Player yardage", "", f"Not scored: {e}", ""])
    if pstats is not None:
        pres = live.score_ledger_players(ledger, games, pstats, cfg["live"]["primary_player_model"])
        if pres.evals:
            psum = SE.summarize_target(pres.evals, v["bootstrap_samples"])
            nmin = min(s["n"] for s in psum.values())
            pnote = (f"`{cfg['live']['primary_player_model']}`, latest pre-kickoff forecast per player, scored exactly "
                     f"from the archived pmf; pool players with no opportunity scored 0 ({pres.no_opportunity}). "
                     f"{pres.pending} game(s) awaiting results or play-by-play. Excluded: {pres.excluded}.")
            if nmin < 200:
                pnote += f" **Small sample ({nmin}+ per stat): not enough to judge accuracy or coverage.**"
            sec = SE.section_lines("player yards", "Prospective", pnote, pres.evals, psum, None, [], v["bootstrap_samples"])
            sec += ["Threshold probabilities P(yards ≥ x): Brier score and mean forecast vs observed rate:", "",
                    "| Stat | x | Forecasts | Mean P | Observed rate | Brier |", "|---|---:|---:|---:|---:|---:|"]
            for stat, by_x in sorted(pres.thresholds.items()):
                for x, pairs in sorted(by_x.items(), key=lambda kv: int(kv[0])):
                    ps = [a for a, _ in pairs]
                    hs = [b for _, b in pairs]
                    br = sum((a - b) ** 2 for a, b in pairs) / len(pairs)
                    sec.append(f"| {stat} | {x} | {len(pairs)} | {sum(ps) / len(ps):.3f} | {sum(hs) / len(hs):.3f} | {br:.4f} |")
            player_sections.append(["## Player yardage", ""] + sec + [""])
            for name, s_ in psum.items():
                print(f"  [players] {name:28s} n={s_['n']:5d} CRPS={s_['crps']:.2f} MAE={s_['mae_median']:.1f}")
        else:
            player_sections.append(["## Player yardage", "", f"No scorable player forecasts yet "
                                    f"({pres.pending} game(s) pending).", ""])
        print(f"Players: {pres.pending} game(s) pending")
    out = PROJECT_ROOT / "reports" / "prospective_scores.md"
    write_markdown(out, title="Prospective scoring (real NFL, issued before kickoff)", data_kind="real",
                   benchmarks=["market_moneyline"],
                   header=[f"Scored against snapshot `{ds.snapshot_id}` ({ds.notes[0]})",
                           f"Ledger: {len(ledger)} forecast records from {len(runs)} run(s)",
                           "`market_at_issue`: market line recorded when our forecast was issued; "
                           "`market_moneyline`: line in the scoring snapshot (likely closing). Market spread/total "
                           "lines are point benchmarks (`*_at_issue`: recorded with our forecast; `*_snapshot`: "
                           "scoring snapshot)."],
                   sections=[section, *dist_sections, *player_sections])
    for name, s in summary.items():
        if s.get("n_scored"):
            print(f"  {name:18s} n={s['n_scored']:4d}  Brier={s['brier']:.4f}  LogLoss={s['log_loss']:.4f}")
    print(f"{res.pending} game(s) pending; excluded {res.excluded}")
    for target, dsum in dist_counts.items():
        for name, s in dsum.items():
            print(f"  [{target}] {name:24s} n={s['n']:4d}  CRPS={s['crps']:.3f}  MAE={s['mae_median']:.2f}")
    print(f"Margin/total: {dres.pending} game(s) pending")
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
    p.add_argument("--live", action="store_true",
                   help="also fetch the current season's play-by-play and injuries (for live player forecasts)")
    p = sub.add_parser("forecast", help="forecast upcoming games and archive them (real data)")
    p.add_argument("--dry-run", action="store_true", help="print forecasts without archiving")
    sub.add_parser("score", help="score archived prospective forecasts against the latest snapshot")
    for name in ("tune", "backtest", "backtest-scores", "backtest-players"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", help="dataset name from config (default: config `dataset`)")
        if name in ("backtest-scores", "backtest-players"):
            p.add_argument("--reveal-holdout", action="store_true", help="also evaluate the sealed holdout (logged)")
        if name == "backtest":
            p.add_argument("--reveal-holdout", action="store_true",
                           help="also evaluate the sealed holdout seasons (logged)")
            p.add_argument("--features", nargs="+", choices=["pbp", "qb"],
                           help="also evaluate feature models: pbp (team play-by-play), qb (QB availability)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if getattr(args, "dataset", None) is None and args.cmd in ("tune", "backtest", "backtest-scores", "backtest-players"):
        args.dataset = cfg["dataset"]
    handlers = {"synth": cmd_synth, "fetch": cmd_fetch, "tune": cmd_tune, "backtest": cmd_backtest,
                "forecast": cmd_forecast, "score": cmd_score, "backtest-scores": cmd_backtest_scores,
                "backtest-players": cmd_backtest_players}
    try:
        return handlers[args.cmd](cfg, args)
    except DatasetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
