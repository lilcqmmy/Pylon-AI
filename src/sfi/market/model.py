"""`market_logit`: logistic regression on the market-implied logit plus the pre-registered indicators.

Targets: home win (market = de-vigged moneyline), home ATS cover (market = de-vigged spread prices), over
(market = de-vigged total prices). Ridge penalty (fixed L2) applies to the indicators only; the intercept and
the market-logit slope are unpenalized, so the model collapses to "market, recalibrated" when the indicators
carry nothing. Walk-forward by season: each season is predicted from a fit on earlier seasons only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sfi.market.dataset import MarketRow

TARGETS = ("win", "cover", "over")
SIDE_FEATURES = ["home_dog", "home_dog_prime", "late_home_dog", "road_fav7", "div_dog_side", "bye_diff",
                 "short_diff", "key_dog_side", "streak_diff"]
OVER_FEATURES = ["dome", "high_total", "primetime"]


def features(x: MarketRow, target: str) -> list[float]:
    f = x.flags
    if target == "over":
        return [0.0 if np.isnan(f["dome"]) else f["dome"], f["high_total"], f["primetime"]]
    dog = 0.0 if x.spread == 0 else (1.0 if x.spread < 0 else -1.0)  # +1 home underdog, -1 away underdog
    return [f["home_dog"], f["home_dog_prime"], f["late_home_dog"], f["road_fav7"], f["div"] * dog,
            f["home_bye_only"] - f["away_bye_only"], f["home_short_only"] - f["away_short_only"],
            f["key"] * dog, f["home_streak3"] - f["away_streak3"]]


def names(target: str) -> list[str]:
    return OVER_FEATURES if target == "over" else SIDE_FEATURES


def market_p(x: MarketRow, target: str) -> float | None:
    return {"win": x.p_home_ml, "cover": x.p_home_cover, "over": x.p_over}[target]


def outcome(x: MarketRow, target: str) -> float | None:
    g = x.game
    if target == "win":
        return None if not g.is_final or g.home_outcome == 0.5 else g.home_outcome
    return x.home_cover() if target == "cover" else x.over()


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def design(rows: list[MarketRow], target: str, with_outcome: bool = True):
    X, y, keep = [], [], []
    for x in rows:
        p = market_p(x, target)
        o = outcome(x, target) if with_outcome else 0.0
        if p is None or o is None:
            continue
        X.append([1.0, _logit(p), *features(x, target)])
        y.append(o)
        keep.append(x)
    return np.asarray(X), np.asarray(y), keep


def fit(X: np.ndarray, y: np.ndarray, l2: float, iters: int = 100) -> np.ndarray:
    pen = np.full(X.shape[1], l2)
    pen[:2] = 0.0  # intercept and market slope unpenalized
    beta = np.zeros(X.shape[1])
    beta[1] = 1.0
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        w = np.maximum(mu * (1 - mu), 1e-9)
        H = X.T @ (X * w[:, None]) + np.diag(pen) + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(H, X.T @ (y - mu) - pen * beta)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))


@dataclass
class WalkForward:
    target: str
    rows: list[MarketRow]
    p_model: np.ndarray
    p_market: np.ndarray
    y: np.ndarray
    blocks: list[str]


def walk_forward(rows: list[MarketRow], target: str, seasons: list[int], l2: float, min_train_seasons: int = 2) -> WalkForward:
    out_rows, pm, pk, ys, bl = [], [], [], [], []
    for s in seasons[min_train_seasons:]:
        Xtr, ytr, _ = design([x for x in rows if x.game.season < s and x.game.season >= seasons[0]], target)
        Xte, yte, kept = design([x for x in rows if x.game.season == s], target)
        if len(ytr) == 0 or len(yte) == 0:
            continue
        beta = fit(Xtr, ytr, l2)
        pm.extend(predict(beta, Xte))
        pk.extend(1 / (1 + np.exp(-Xte[:, 1])))
        ys.extend(yte)
        out_rows.extend(kept)
        bl.extend(f"{x.game.season}-W{x.game.week:02d}" for x in kept)
    return WalkForward(target, out_rows, np.asarray(pm), np.asarray(pk), np.asarray(ys), bl)


def bootstrap_coefficients(rows: list[MarketRow], target: str, l2: float, n_boot: int = 400, seed: int = 0):
    """Fit on all rows; percentile 95% CI from resampling whole weeks."""
    X, y, kept = design(rows, target)
    beta = fit(X, y, l2)
    blocks = np.array([f"{x.game.season}-W{x.game.week:02d}" for x in kept])
    uniq, inv = np.unique(blocks, return_inverse=True)
    idx_by = [np.nonzero(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[i] for i in pick])
        draws.append(fit(X[idx], y[idx], l2))
    draws = np.asarray(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return beta, lo, hi, len(y)


def market_calibration(rows: list[MarketRow], target: str, n_boot: int = 400, seed: int = 0):
    """Recalibration of the market alone: outcome ~ a + b * logit(market). Perfect: a = 0, b = 1."""
    X, y, kept = design(rows, target)
    X2 = X[:, :2]
    beta = fit(X2, y, 0.0)
    blocks = np.array([f"{x.game.season}-W{x.game.week:02d}" for x in kept])
    uniq, inv = np.unique(blocks, return_inverse=True)
    idx_by = [np.nonzero(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[i] for i in pick])
        draws.append(fit(X2[idx], y[idx], 0.0))
    draws = np.asarray(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return beta, lo, hi, len(y)
