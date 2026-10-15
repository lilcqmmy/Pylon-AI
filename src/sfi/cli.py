"""Command line: `sfi synth` generates fictional data, `sfi backtest` runs the walk-forward baseline."""

from __future__ import annotations

import argparse
import json
import sys

from sfi.config import PROJECT_ROOT, load_config
from sfi.data import synthetic
from sfi.data.schema import file_sha256, read_games_csv, write_games_csv
from sfi.evaluation.report import ORACLE, score, summarize, write_markdown
from sfi.evaluation.walkforward import GateConfig, run_walkforward
from sfi.forecasts.record import ForecastArchive
from sfi.models.baselines import CoinFlip, HomeRate
from sfi.models.elo import EloModel, EloParams

SYN_DIR = PROJECT_ROOT / "data" / "synthetic"
GAMES_CSV = SYN_DIR / "games.csv"
TRUTH_CSV = SYN_DIR / "truth_DO_NOT_USE_AS_FEATURE.csv"


def cmd_synth(cfg: dict) -> int:
    s = cfg["synthetic"]
    games, truth = synthetic.generate(s["seed"], s["seasons"], s["n_teams"], s["weeks_per_season"])
    write_games_csv(games, GAMES_CSV)
    synthetic.write_truth_csv(truth, TRUTH_CSV)
    print(f"Wrote {len(games)} synthetic games to {GAMES_CSV.relative_to(PROJECT_ROOT)}")
    print(f"sha256 {file_sha256(GAMES_CSV)}")
    return 0


def cmd_backtest(cfg: dict) -> int:
    if not GAMES_CSV.exists():
        print("No synthetic data yet; run `sfi synth` first.", file=sys.stderr)
        return 1
    games = read_games_csv(GAMES_CSV)
    snapshot_id = "sha256:" + file_sha256(GAMES_CSV)[:16]
    elo_cfg = cfg["elo"]
    params = EloParams(**{k: elo_cfg[k] for k in EloParams.__dataclass_fields__})
    gate = GateConfig(cfg["forecast"]["publish_gate"].get("min_prior_games_per_team") or 0)

    records = run_walkforward(
        games, [lambda: EloModel(params), HomeRate, CoinFlip],
        source_snapshot_id=snapshot_id, data_kind="synthetic", gate=gate,
    )
    archive = ForecastArchive(PROJECT_ROOT / cfg["storage"]["backtest_archive"])
    written, same = archive.append(records)
    print(f"Archive: {written} new records, {same} identical re-runs ({len(archive)} total)")

    v = cfg["validation"]
    seasons = sorted({g.season for g in games})
    burn_in = set(seasons[: v["burn_in_seasons"]])
    truth = {gid: t.p_home_win_given_no_tie for gid, t in synthetic.read_truth_csv(TRUTH_CSV).items()}
    sets = score(records, {g.game_id: g for g in games}, burn_in, oracle=truth)
    summary = summarize(sets, v["bootstrap_samples"])

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "baseline_synthetic.json").write_text(json.dumps(
        {"data_kind": "synthetic", "snapshot_id": snapshot_id, "burn_in_seasons": sorted(burn_in),
         "elo_params": elo_cfg, "summary": summary}, indent=2), encoding="utf-8")
    write_markdown(
        reports / "baseline_synthetic.md", title="Baseline walk-forward evaluation (synthetic)",
        data_kind="synthetic", snapshot_id=snapshot_id,
        config_note=f"Scored seasons: {[s for s in seasons if s not in burn_in]} "
                    f"(burn-in: {sorted(burn_in)}); Elo params: {elo_cfg}",
        summary=summary, sets=sets, primary="baseline_elo",
        comparisons=[("baseline_elo", "home_rate"), ("baseline_elo", "coin_flip"),
                     ("baseline_elo", ORACLE)],
        n_boot=v["bootstrap_samples"], n_bins=v["calibration_bins"],
    )
    for name, s in summary.items():
        if s.get("n_scored"):
            print(f"{name:18s} n={s['n_scored']:4d}  Brier={s['brier']:.4f}  LogLoss={s['log_loss']:.4f}")
    print("Report: reports/baseline_synthetic.md  (SYNTHETIC — not real NFL results)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sfi", description=__doc__)
    ap.add_argument("--config", help="path to YAML config (default: config/config.yaml or example)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("synth", help="generate fictional NFL-like seasons")
    sub.add_parser("backtest", help="walk-forward baseline backtest + report")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    return {"synth": cmd_synth, "backtest": cmd_backtest}[args.cmd](cfg)


if __name__ == "__main__":
    raise SystemExit(main())
