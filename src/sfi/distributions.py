"""Discrete predictive distributions over integer outcomes (score margin, game total).

NFL margins and totals are integers with heavy mass on key numbers (3, 7, ...), so forecasts are
stored as a probability mass function over an integer range. Scoring rules are exact for that support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiscreteDist:
    lo: int            # support is lo, lo+1, ..., lo+len(pmf)-1
    pmf: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.pmf, dtype=float)
        if p.ndim != 1 or len(p) == 0 or np.any(p < -1e-12):
            raise ValueError("pmf must be a non-empty 1-D array of non-negative values")
        s = p.sum()
        if not s > 0:
            raise ValueError("pmf has no mass")
        object.__setattr__(self, "pmf", np.clip(p, 0, None) / s)

    @property
    def support(self) -> np.ndarray:
        return np.arange(self.lo, self.lo + len(self.pmf))

    @property
    def cdf(self) -> np.ndarray:
        return np.cumsum(self.pmf)

    def mean(self) -> float:
        return float(self.support @ self.pmf)

    def std(self) -> float:
        m = self.mean()
        return float(math.sqrt(((self.support - m) ** 2) @ self.pmf))

    def prob(self, k: int) -> float:
        i = k - self.lo
        return float(self.pmf[i]) if 0 <= i < len(self.pmf) else 0.0

    def cdf_at(self, k: float) -> float:
        """P(Y <= k)."""
        i = math.floor(k) - self.lo
        if i < 0:
            return 0.0
        if i >= len(self.pmf):
            return 1.0
        return float(self.cdf[i])

    def quantile(self, q: float) -> int:
        """Smallest k with P(Y <= k) >= q."""
        if not 0 < q < 1:
            raise ValueError("q must be in (0, 1)")
        i = int(np.searchsorted(self.cdf, q - 1e-12))
        return int(self.lo + min(i, len(self.pmf) - 1))

    def central_interval(self, level: float) -> tuple[int, int]:
        a = (1 - level) / 2
        return self.quantile(a), self.quantile(1 - a)

    def mass(self, a: int, b: int) -> float:
        """P(a <= Y <= b)."""
        return self.cdf_at(b) - self.cdf_at(a - 1)

    def p_greater(self, k: float) -> float:
        return 1.0 - self.cdf_at(k)

    # ---- scoring rules (lower is better)
    def crps(self, y: int) -> float:
        """Exact CRPS on the integers: sum over k of (F(k) - 1{y <= k})^2, including integers between
        the outcome and the support when the outcome falls outside it."""
        ind = (self.support >= y).astype(float)
        hi = self.lo + len(self.pmf) - 1
        tail = 0.0
        if y < self.lo:   # k in [y, lo-1]: F = 0, indicator 1
            tail = float(self.lo - y)
        elif y > hi:      # k in [hi+1, y-1]: F = 1, indicator 0
            tail = float(y - hi - 1)
        return float(((self.cdf - ind) ** 2).sum()) + tail

    def pinball(self, y: int, qs: tuple[float, ...]) -> float:
        total = 0.0
        for q in qs:
            x = self.quantile(q)
            total += max(q * (y - x), (q - 1) * (y - x))
        return total / len(qs)

    def pit(self, y: int, u: float) -> float:
        """Randomized probability integral transform: uniform on (0, 1) if the forecast is calibrated."""
        lo_f = self.cdf_at(y - 1)
        return lo_f + u * (self.cdf_at(y) - lo_f)

    def summary(self, qs: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)) -> dict:
        return {"mean": round(self.mean(), 4), "sd": round(self.std(), 4),
                **{f"q{int(round(q * 100)):02d}": self.quantile(q) for q in qs}}


def from_normal(mu: float, sigma: float, lo: int, hi: int) -> DiscreteDist:
    """Normal discretized to integers (mass of [k - 0.5, k + 0.5]), truncated to [lo, hi]."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    edges = np.arange(lo, hi + 2) - 0.5
    z = (edges - mu) / (sigma * math.sqrt(2))
    cdf = 0.5 * (1 + np.vectorize(math.erf)(z))
    pmf = np.diff(cdf)
    if pmf.sum() <= 0:
        pmf = np.zeros(hi - lo + 1)
        pmf[int(np.clip(round(mu), lo, hi)) - lo] = 1.0
    return DiscreteDist(lo, pmf)


def from_residuals(mu: float, residuals: np.ndarray, lo: int, hi: int) -> DiscreteDist:
    """Distribution of round(mu + r) over past residuals r, truncated to [lo, hi]. Keeps the shape
    of past errors (including mass on key numbers relative to the mean)."""
    vals = np.clip(np.rint(mu + np.asarray(residuals, dtype=float)), lo, hi).astype(int)
    pmf = np.bincount(vals - lo, minlength=hi - lo + 1).astype(float)
    return DiscreteDist(lo, pmf)


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
