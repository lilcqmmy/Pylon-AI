"""Prospective (live) forecasting and scoring.

`sfi forecast` replays all history exactly as the backtest does, advances to the wall-clock time,
and forecasts upcoming games. Forecasts go to an append-only ledger together with a run record: data
snapshot, code commit, tuned-parameter file hash, and the market line at issue time (benchmark only).

Refuses to run on stale data: a game whose result should already be known but has no score means
the ratings would be missing information.

`sfi score` scores ledger forecasts once results arrive in a newer snapshot. Per (model, game) it
scores the **latest forecast issued before the actual kickoff**; forecasts issued after a game's
(possibly rescheduled) kickoff are excluded.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from sfi.data.schema import Game
from sfi.evaluation.report import ScoredSet
from sfi.evaluation.scores_eval import ScoreForecast, make_score_record
from sfi.evaluation.players_eval import make_player_record
from sfi.evaluation.walkforward import Forecast, GateConfig, Replay, slates_by_kickoff
from sfi.forecasts.record import ForecastRecord, make_record
from sfi.models.base import WinModel


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class LiveConfig:
    horizon_days: float = 8.0
    min_lead_minutes: float = 10.0
    max_snapshot_age_hours: float = 24.0
    known_at_hours: float = 5.0


class StaleDataError(RuntimeError):
    pass


def freshness_problems(games: list[Game], retrieved_at: datetime, now: datetime, cfg: LiveConfig) -> list[str]:
    problems = []
    age = now - retrieved_at
    if age > timedelta(hours=cfg.max_snapshot_age_hours):
        problems.append(f"snapshot is {age.total_seconds() / 3600:.1f}h old (max {cfg.max_snapshot_age_hours:g}h)")
    overdue = sorted(
        (g for g in games if not g.is_final
         and g.scheduled_start_utc + timedelta(hours=cfg.known_at_hours) <= now),
        key=lambda g: g.scheduled_start_utc,
    )
    if overdue:
        ids = ", ".join(g.game_id for g in overdue[:8])
        problems.append(f"{len(overdue)} game(s) should have a result by now but have none: {ids}")
    if retrieved_at > now:
        problems.append("snapshot retrieval time is in the future")
    return problems


def select_targets(games: list[Game], now: datetime, cfg: LiveConfig) -> list[Game]:
    lo = now + timedelta(minutes=cfg.min_lead_minutes)
    hi = now + timedelta(days=cfg.horizon_days)
    return sorted((g for g in games if not g.is_final and lo <= g.scheduled_start_utc <= hi),
                  key=lambda g: (g.scheduled_start_utc, g.game_id))


def _replay_to_now(games: list[Game], model, now: datetime, targets: list[Game], as_of_lead: timedelta,
                   issue: Callable[[Replay, Game], object]) -> Replay:
    """Replay history slates whose as-of time has passed, issuing forecasts exactly as the backtest
    would (models with walk-forward calibration learn from them), then advance to `now`."""
    target_ids = {g.game_id for g in targets}
    replay = Replay(games, model)
    for kickoff, slate in slates_by_kickoff([g for g in games if g.game_id not in target_ids]):
        as_of = kickoff - as_of_lead
        if as_of > now:
            break
        replay.advance_to(as_of)
        for g in slate:
            issue(replay, g)
    replay.advance_to(now)
    return replay


def forecast_now(
    games: list[Game], model: WinModel, now: datetime, targets: list[Game], *,
    as_of_lead: timedelta, gate: GateConfig = GateConfig(),
) -> list[Forecast]:
    """Win-probability forecasts for `targets` issued at `now`. Only results known by `now` reach the model."""
    replay = _replay_to_now(games, model, now, targets, as_of_lead,
                            lambda r, g: r.forecast(g, gate, with_inputs=False))
    return [replay.forecast(g, gate) for g in targets]


def score_forecast_now(games: list[Game], model, now: datetime, targets: list[Game], *,
                       as_of_lead: timedelta) -> list[ScoreForecast]:
    """Margin/total distributions for `targets` issued at `now` (same replay rules as `forecast_now`)."""
    def issue(r: Replay, g: Game):
        r.prepare(g)
        model.predict_scores(g)

    replay = _replay_to_now(games, model, now, targets, as_of_lead, issue)
    out = []
    for g in targets:
        replay.prepare(g)
        m, t = model.predict_scores(g)
        out.append(ScoreForecast(g, replay.as_of, m, t, model.inputs_for(g)))
    return out


def player_forecast_now(games: list[Game], model, now: datetime, targets: list[Game], *,
                        as_of_lead: timedelta) -> list[tuple[datetime, object]]:
    """Player yardage distributions for `targets` issued at `now` (same replay rules as `forecast_now`)."""
    def issue(r: Replay, g: Game):
        r.prepare(g)
        model.predict_players(g)

    replay = _replay_to_now(games, model, now, targets, as_of_lead, issue)
    out = []
    for g in targets:
        replay.prepare(g)
        out += [(replay.as_of, f) for f in model.predict_players(g)]
    return out


def pbp_gaps(games: list[Game], player_stats: dict, now: datetime, season: int) -> list[Game]:
    """Final games of `season` whose results are known by `now` but have no play-by-play player rows."""
    return [g for g in games if g.season == season and g.is_final and g.result_known_by(now)
            and g.game_id not in player_stats]


def git_state(root: Path) -> tuple[str | None, bool | None]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def build_run(
    *, games: list[Game], factories: list[Callable[[], WinModel]], now: datetime, cfg: LiveConfig,
    as_of_lead: timedelta, gate: GateConfig, snapshot_id: str, manifest: dict,
    market: dict[str, float], extra: dict, score_factories: list[Callable[[], object]] = (),
    lines: dict[str, dict] | None = None, player_factory: Callable[[], object] | None = None,
    player_stats: dict | None = None, thresholds: dict[str, list[int]] | None = None,
    max_pbp_gap_hours: float = 48.0,
) -> tuple[list[ForecastRecord], dict]:
    retrieved = _parse(manifest["retrieved_at_utc"])
    problems = freshness_problems(games, retrieved, now, cfg)
    if problems:
        raise StaleDataError("; ".join(problems))
    targets = select_targets(games, now, cfg)
    records: list[ForecastRecord] = []
    for factory in factories:
        model = factory()
        for f in forecast_now(games, model, now, targets, as_of_lead=as_of_lead, gate=gate):
            records.append(make_record(
                game=f.game, as_of=f.as_of, p_home_win=f.p_home_win, model_name=model.name,
                model_version=model.version, feature_spec_version=model.feature_spec_version,
                source_snapshot_id=snapshot_id, inputs=f.inputs, gate=f.gate, data_kind="real",
            ))
    for factory in score_factories:
        model = factory()
        for f in score_forecast_now(games, model, now, targets, as_of_lead=as_of_lead):
            records.append(make_score_record(f, model, source_snapshot_id=snapshot_id, data_kind="real",
                                             include_pmf=True))
    player_status = None
    if player_factory is not None:
        season = max((g.season for g in targets), default=None)
        gaps = pbp_gaps(games, player_stats or {}, now, season) if season is not None else []
        old = [g for g in gaps if now - g.result_known_at_utc > timedelta(hours=max_pbp_gap_hours)]
        if old:
            # Play-by-play is badly behind: player histories would silently miss games. Skip players.
            player_status = {"skipped": True, "reason": f"play-by-play missing for {len(old)} game(s) final for "
                             f"more than {max_pbp_gap_hours:g}h", "games": [g.game_id for g in old]}
        else:
            model = player_factory()
            for as_of, f in player_forecast_now(games, model, now, targets, as_of_lead=as_of_lead):
                records.append(make_player_record(as_of, f, model, source_snapshot_id=snapshot_id, data_kind="real",
                                                  include_pmf=True, thresholds=thresholds))
            player_status = {"skipped": False, "recent_pbp_gaps": [g.game_id for g in gaps]}
    issued = _iso(now)
    lines = lines or {}
    run = {
        "run_id": hashlib.sha256(f"{issued}|{snapshot_id}".encode()).hexdigest()[:12],
        "issued_at_utc": issued, "snapshot_id": snapshot_id,
        "snapshot_retrieved_at_utc": manifest["retrieved_at_utc"],
        "upstream_last_updated": manifest.get("upstream_last_updated"),
        "models": sorted({f"{r.model_name}@{r.model_version}" for r in records}),
        "game_ids": [g.game_id for g in targets], "n_records": len(records),
        "market_p_home_at_issue": {g.game_id: round(market[g.game_id], 6) for g in targets if g.game_id in market},
        "market_lines_at_issue": {g.game_id: lines[g.game_id] for g in targets if g.game_id in lines},
        **({"players": player_status} if player_status is not None else {}),
        **extra,
    }
    return records, run


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass
class ProspectiveScore:
    sets: dict[str, ScoredSet]
    excluded: dict[str, int]
    pending: int


def _latest_pre_kickoff(ledger: list[dict], games: dict[str, Game], outcome_type: str,
                        excluded: dict[str, int]) -> dict[tuple[str, str], dict]:
    """Per (model, game): the latest record of `outcome_type` issued before the actual kickoff."""
    latest: dict[tuple[str, str], dict] = {}
    for r in ledger:
        if r.get("outcome_type", "home_win_excl_tie") != outcome_type:
            continue
        g = games.get(r["game_id"])
        if g is None:
            excluded["game_not_in_snapshot"] += 1
            continue
        if _parse(r["prediction_as_of_utc"]) >= g.scheduled_start_utc:
            excluded["issued_at_or_after_kickoff"] += 1
            continue
        key = (r["model_name"], r["game_id"])
        if key not in latest or r["prediction_as_of_utc"] > latest[key]["prediction_as_of_utc"]:
            latest[key] = r
    return latest


@dataclass
class ProspectiveScoreDist:
    evals: dict[str, dict]       # target -> {model: TargetEval}
    market: dict[str, dict]      # target -> {"at_issue": TargetEval, "snapshot": TargetEval}
    excluded: dict[str, int]
    pending: int


def score_ledger_distributions(
    ledger: list[dict], runs: list[dict], games: dict[str, Game], lines_now: dict[str, dict],
) -> ProspectiveScoreDist:
    """Margin/total forecasts: exact CRPS etc. from the stored pmfs, latest pre-kickoff record per model."""
    from sfi.evaluation.scores_eval import TargetEval, evaluate, pmf_from_json

    excluded = {"game_not_in_snapshot": 0, "issued_at_or_after_kickoff": 0, "no_pmf": 0}
    latest = _latest_pre_kickoff(ledger, games, "home_margin_and_total", excluded)
    fcs, pending, used = [], set(), {}
    for (model, gid), r in latest.items():
        g = games[gid]
        if not g.is_final:
            pending.add(gid)
            continue
        d = r["distribution"]
        if "margin_pmf" not in d:
            excluded["no_pmf"] += 1
            continue
        fcs.append((model, ScoreForecast(g, _parse(r["prediction_as_of_utc"]), pmf_from_json(d["margin_pmf"]),
                                         pmf_from_json(d["total_pmf"]), {})))
        used[gid] = max(used.get(gid, ""), r["prediction_as_of_utc"])
    seasons = {games[gid].season for gid in used}
    evals = {t: evaluate(fcs, t, seasons) for t in ("margin", "total")}
    # Market lines: as recorded by the run that issued the scored forecast, and in the scoring snapshot.
    at_issue_lines: dict[str, dict] = {}
    by_time = {r["issued_at_utc"]: r.get("market_lines_at_issue", {}) for r in runs}
    for gid, t in used.items():
        if gid in by_time.get(t, {}):
            at_issue_lines[gid] = by_time[t][gid]
    market = {}
    for target in ("margin", "total"):
        key = "spread" if target == "margin" else "total"
        m = {}
        for label, src in (("at_issue", at_issue_lines), ("snapshot", {k: v for k, v in lines_now.items() if k in used})):
            e = TargetEval(f"market_{key}_{label}")
            for gid, ln in src.items():
                g = games[gid]
                if ln.get(key) is None:
                    continue
                y = g.home_score - g.away_score if target == "margin" else g.home_score + g.away_score
                e.game_ids.append(gid)
                e.blocks.append(f"{g.season}-W{g.week:02d}")
                e.y.append(y)
                e.abs_err_median.append(abs(y - ln[key]))
                e.sq_err_mean.append((y - ln[key]) ** 2)
            m[label] = e
        market[target] = m
    return ProspectiveScoreDist(evals, market, excluded, len(pending))


@dataclass
class ProspectivePlayers:
    evals: dict[str, object]                  # stat -> TargetEval (primary model)
    thresholds: dict[str, dict[str, list]]    # stat -> threshold -> [(p, hit)]
    no_opportunity: dict[str, int]
    excluded: dict[str, int]
    pending: int


def score_ledger_players(ledger: list[dict], games: dict[str, Game], player_stats: dict,
                         model_name: str, seed: int = 0) -> ProspectivePlayers:
    """Latest pre-kickoff forecast per (game, player, stat) for `model_name`, scored exactly from its pmf.
    A game is scored only once it is final and its play-by-play is available; pool players with no
    opportunity score 0 yards."""
    import numpy as np

    from sfi.evaluation.scores_eval import LEVELS, PINBALL_QS, TargetEval, pmf_from_json

    excluded = {"game_not_in_snapshot": 0, "issued_at_or_after_kickoff": 0}
    latest: dict[tuple, dict] = {}
    for r in ledger:
        ot = r.get("outcome_type", "")
        if not ot.startswith("player_") or r["model_name"] != model_name:
            continue
        g = games.get(r["game_id"])
        if g is None:
            excluded["game_not_in_snapshot"] += 1
            continue
        if _parse(r["prediction_as_of_utc"]) >= g.scheduled_start_utc:
            excluded["issued_at_or_after_kickoff"] += 1
            continue
        key = (r["game_id"], r["distribution"]["player"], ot)
        if key not in latest or r["prediction_as_of_utc"] > latest[key]["prediction_as_of_utc"]:
            latest[key] = r
    got = {(gid, p, s): y for gid, rows in player_stats.items() for p, _, s, _, y in rows}
    rng = np.random.default_rng(seed)
    evals, thr, zeros, pending = {}, {}, {}, set()
    for (gid, player, ot), r in sorted(latest.items()):
        g = games[gid]
        if not g.is_final or gid not in player_stats:
            pending.add(gid)
            continue
        stat = ot.split("_")[1]
        d = pmf_from_json(r["distribution"]["pmf"])
        y = got.get((gid, player, stat))
        if y is None:
            zeros[stat] = zeros.get(stat, 0) + 1
            y = 0.0
        y = int(round(y))
        e = evals.setdefault(stat, TargetEval(f"{model_name}:{stat}"))
        e.game_ids.append(f"{gid}|{player}|{stat}")
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
        for x, pr in r["distribution"].get("p_at_least", {}).items():
            thr.setdefault(stat, {}).setdefault(x, []).append((pr, float(y >= int(x))))
    return ProspectivePlayers(evals, thr, zeros, excluded, len(pending))


def score_ledger(
    ledger: list[dict], games: dict[str, Game], benchmarks: dict[str, dict[str, float]] | None = None,
) -> ProspectiveScore:
    excluded = {"game_not_in_snapshot": 0, "issued_at_or_after_kickoff": 0, "tie": 0, "abstained": 0}
    pending_games: set[str] = set()
    latest = _latest_pre_kickoff(ledger, games, "home_win_excl_tie", excluded)
    sets: dict[str, ScoredSet] = {}
    for (model, gid), r in sorted(latest.items(), key=lambda kv: (kv[0][0], games[kv[0][1]].scheduled_start_utc)):
        g = games[gid]
        s = sets.setdefault(model, ScoredSet(model))
        if not g.is_final:
            pending_games.add(gid)
        elif r["quality_gate"]["status"] != "publish":
            s.n_abstain += 1
        elif g.home_outcome == 0.5:
            s.n_tie += 1
        else:
            s.game_ids.append(gid)
            s.p.append(r["distribution"]["p_home_win"])
            s.y.append(g.home_outcome)
            s.block.append(f"{g.season}-W{g.week:02d}")
    if benchmarks and sets:
        ref = max(sets.values(), key=lambda s: len(s.p))
        for name, probs in benchmarks.items():
            o = ScoredSet(name)
            for gid, y, b in zip(ref.game_ids, ref.y, ref.block):
                if gid in probs:
                    o.game_ids.append(gid)
                    o.p.append(probs[gid])
                    o.y.append(y)
                    o.block.append(b)
            sets[name] = o
    return ProspectiveScore(sets, excluded, len(pending_games))


def market_at_issue(runs: list[dict], ledger: list[dict], games: dict[str, Game]) -> dict[str, float]:
    """For each game, the market line recorded in the run that issued the latest pre-kickoff forecast."""
    by_run_time = {r["issued_at_utc"]: r.get("market_p_home_at_issue", {}) for r in runs}
    out: dict[str, tuple[str, float]] = {}
    for rec in ledger:
        gid, t = rec["game_id"], rec["prediction_as_of_utc"]
        if gid not in games or _parse(t) >= games[gid].scheduled_start_utc:
            continue
        p = by_run_time.get(t, {}).get(gid)
        if p is not None and (gid not in out or t > out[gid][0]):
            out[gid] = (t, p)
    return {gid: p for gid, (_, p) in out.items()}
