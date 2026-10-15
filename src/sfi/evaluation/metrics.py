"""Proper scoring rules, calibration diagnostics and time-blocked bootstrap intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-15


def _arrays(p, y) -> tuple[np.ndarray, np.ndarray]:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    if p.shape != y.shape or p.ndim != 1 or len(p) == 0:
        raise ValueError("p and y must be equal-length, non-empty 1-D arrays")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("y must be binary 0/1 (exclude ties before scoring)")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p must be in [0, 1]")
    return p, y


def brier_terms(p, y) -> np.ndarray:
    p, y = _arrays(p, y)
    return (p - y) ** 2


def log_loss_terms(p, y) -> np.ndarray:
    p, y = _arrays(p, y)
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(p, y) -> float:
    return float(brier_terms(p, y).mean())


def log_loss(p, y) -> float:
    return float(log_loss_terms(p, y).mean())


@dataclass(frozen=True)
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_p: float
    observed: float


def calibration_table(p, y, n_bins: int = 10) -> list[CalibrationBin]:
    p, y = _arrays(p, y)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append(CalibrationBin(edges[b], edges[b + 1], int(m.sum()),
                                      float(p[m].mean()), float(y[m].mean())))
    return out


def calibration_intercept_slope(p, y, iters: int = 50) -> tuple[float, float]:
    """Logistic recalibration y ~ a + b*logit(p). Perfect calibration: a=0, b=1.

    Returns (nan, nan) when forecasts barely vary: the slope is not identifiable then.
    """
    p, y = _arrays(p, y)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    if np.std(logit) < 0.05:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones_like(p), logit])
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ beta))
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def block_bootstrap_mean_ci(
    values, blocks, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of `values` with a CI from resampling whole blocks (e.g. weeks) with replacement."""
    values = np.asarray(values, dtype=float)
    blocks = np.asarray(blocks)
    uniq, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)
