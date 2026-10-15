"""Seeded, entirely fictional NFL-like seasons for testing the pipeline.

Teams are "T01".."T32"; nothing here is real data. Each team has a hidden true strength (in points)
that drifts week to week. Scores are 7*Poisson + 3*Poisson, so ties occur occasionally.

The hidden truth (exact win probability per game) is written to a separate file and must never be
used as a model input; it exists only to measure how far a model is from the best possible forecast.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from sfi.data.schema import Game

LEAGUE_MEAN_POINTS = 22.0
HOME_ADVANTAGE_POINTS = 2.0
TD_SHARE = 0.7  # share of expected points from 7-point scores; the rest from 3-point scores
MAX_POINTS = 140
SOURCE_NAME = "synthetic_v1"


@dataclass(frozen=True)
class TruthRow:
    game_id: str
    p_home_win: float
    p_tie: float

    @property
    def p_home_win_given_no_tie(self) -> float:
        return self.p_home_win / (1.0 - self.p_tie)


def _poisson_pmf(lam: float, n: int) -> np.ndarray:
    k = np.arange(n + 1)
    logp = k * np.log(lam) - lam - np.cumsum(np.log(np.maximum(k, 1)))
    return np.exp(logp)


def points_pmf(mean_points: float) -> np.ndarray:
    """Exact pmf of 7*Poisson(a) + 3*Poisson(b) truncated at MAX_POINTS."""
    lam_td = mean_points * TD_SHARE / 7.0
    lam_fg = mean_points * (1 - TD_SHARE) / 3.0
    td = _poisson_pmf(lam_td, MAX_POINTS // 7)
    fg = _poisson_pmf(lam_fg, MAX_POINTS // 3)
    pmf = np.zeros(MAX_POINTS + 1)
    for i, pi in enumerate(td):
        pts = 7 * i + 3 * np.arange(len(fg))
        ok = pts <= MAX_POINTS
        np.add.at(pmf, pts[ok], pi * fg[ok])
    return pmf / pmf.sum()


def outcome_probs(home_mean: float, away_mean: float) -> tuple[float, float]:
    """(P(home wins), P(tie)) under the synthetic scoring model."""
    ph, pa = points_pmf(home_mean), points_pmf(away_mean)
    cdf_a = np.cumsum(pa)
    p_win = float(np.sum(ph[1:] * cdf_a[:-1]))
    p_tie = float(np.sum(ph * pa))
    return p_win, p_tie


def expected_points(home_strength: float, away_strength: float) -> tuple[float, float]:
    margin = home_strength - away_strength + HOME_ADVANTAGE_POINTS
    return max(LEAGUE_MEAN_POINTS + margin / 2, 3.0), max(LEAGUE_MEAN_POINTS - margin / 2, 3.0)


def generate(
    seed: int, seasons: list[int], n_teams: int = 32, weeks_per_season: int = 17
) -> tuple[list[Game], list[TruthRow]]:
    if n_teams % 2:
        raise ValueError("n_teams must be even")
    rng = np.random.default_rng(seed)
    teams = [f"T{i + 1:02d}" for i in range(n_teams)]
    strength = rng.normal(0.0, 6.0, n_teams)
    games: list[Game] = []
    truth: list[TruthRow] = []
    for s_idx, season in enumerate(seasons):
        if s_idx > 0:
            strength = 0.7 * strength + rng.normal(0.0, 3.0, n_teams)
        season_start = datetime(season, 9, 8, 17, 0, tzinfo=timezone.utc)
        for week in range(1, weeks_per_season + 1):
            strength = strength + rng.normal(0.0, 0.5, n_teams)
            order = rng.permutation(n_teams)
            week_start = season_start + timedelta(days=7 * (week - 1))
            for slot, (h, a) in enumerate(zip(order[0::2], order[1::2])):
                hm, am = expected_points(strength[h], strength[a])
                td_h, fg_h = rng.poisson(hm * TD_SHARE / 7), rng.poisson(hm * (1 - TD_SHARE) / 3)
                td_a, fg_a = rng.poisson(am * TD_SHARE / 7), rng.poisson(am * (1 - TD_SHARE) / 3)
                kickoff = week_start + timedelta(hours=3 * (slot % 3))
                gid = f"SYN-{season}-W{week:02d}-{slot + 1:02d}"
                games.append(Game(
                    league="NFL", season=season, week=week, game_id=gid,
                    home_team=teams[h], away_team=teams[a], scheduled_start_utc=kickoff,
                    home_score=int(7 * td_h + 3 * fg_h), away_score=int(7 * td_a + 3 * fg_a),
                    result_known_at_utc=kickoff + timedelta(hours=4), source_name=SOURCE_NAME,
                ))
                p_win, p_tie = outcome_probs(hm, am)
                truth.append(TruthRow(gid, p_win, p_tie))
    return games, truth


def write_truth_csv(truth: list[TruthRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["game_id", "p_home_win", "p_tie"])
        for t in truth:
            w.writerow([t.game_id, f"{t.p_home_win:.6f}", f"{t.p_tie:.6f}"])


def read_truth_csv(path: str | Path) -> dict[str, TruthRow]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {
            r["game_id"]: TruthRow(r["game_id"], float(r["p_home_win"]), float(r["p_tie"]))
            for r in csv.DictReader(fh)
        }
