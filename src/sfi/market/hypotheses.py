"""Tests of the pre-registered market-bias hypotheses (docs/market_biases.md)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sfi.evaluation.metrics import block_bootstrap_mean_ci
from sfi.market.dataset import MarketRow
from sfi.picks import american_profit

VALIDATION_FAMILY = ["a1", "a2", "j", "b", "c2", "i", "d", "e1", "f", "k"]  # e2 dropped: Amendment 1
Z975, Z_POWER = 1.959964, 0.841621


# ------------------------------------------------------------------ exact binomial tools
def _log_pmf(k: int, n: int, p: float) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log(1 - p)


def binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided p: total probability of outcomes no more likely than the observed one."""
    if n == 0:
        return 1.0
    lp = [_log_pmf(i, n, p) for i in range(n + 1)]
    obs = lp[k]
    return min(1.0, sum(math.exp(v) for v in lp if v <= obs + 1e-9))


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    return min(1.0, sum(math.exp(_log_pmf(i, n, p)) for i in range(0, k + 1)))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)

    def solve(f, lo=0.0, hi=1.0):
        for _ in range(80):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - _binom_cdf(k - 1, n, p) >= alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p) <= alpha / 2)
    return lower, upper


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, tuple[float, float]]:
    """(difference p1 - p2, two-sided p, 95% CI) via the normal approximation."""
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se0 = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se0 if se0 > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return p1 - p2, p, (p1 - p2 - Z975 * se, p1 - p2 + Z975 * se)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=lambda k: pvals[k])
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def mde_rate(n: int, z: float = Z975) -> float:
    return (z + Z_POWER) * math.sqrt(0.25 / n) if n else float("nan")


# ------------------------------------------------------------------ per-hypothesis selections
@dataclass
class Bets:
    hits: list[float] = field(default_factory=list)      # 1/0, pushes excluded
    returns: list[float] = field(default_factory=list)   # per unit, pushes 0
    blocks: list[str] = field(default_factory=list)

    def add(self, hit: float | None, odds: float, block: str) -> None:
        if hit is None:
            self.returns.append(0.0)
        else:
            self.hits.append(hit)
            self.returns.append(american_profit(odds, 1.0) if hit == 1.0 else -1.0)
        self.blocks.append(block)


def _block(x: MarketRow) -> str:
    return f"{x.game.season}-W{x.game.week:02d}"


def _home_side(x: MarketRow, bets: Bets) -> None:
    hc = x.home_cover()
    bets.add(hc, x.home_spread_odds, _block(x))


def _side(x: MarketRow, side: str, bets: Bets) -> None:
    hc = x.home_cover()
    hit = None if hc is None else (hc if side == "home" else 1.0 - hc)
    bets.add(hit, x.home_spread_odds if side == "home" else x.away_spread_odds, _block(x))


def _underdog(x: MarketRow) -> str | None:
    return None if x.spread == 0 else ("home" if x.spread < 0 else "away")


def select(rows: list[MarketRow], hid: str) -> dict[str, Bets]:
    """Bets on the hypothesized side; for difference hypotheses, two groups."""
    final = [x for x in rows if x.game.is_final]
    if hid in ("a1", "a2", "j", "b"):
        flag = {"a1": "home_dog", "a2": "home_dog_prime", "j": "late_home_dog", "b": "road_fav7"}[hid]
        b = Bets()
        for x in final:
            if x.flags[flag]:
                _home_side(x, b)
        return {"main": b}
    if hid == "i":
        b = Bets()
        for x in final:
            if x.flags["high_total"]:
                o = x.over()
                b.add(None if o is None else 1.0 - o, x.under_odds, _block(x))
        return {"main": b}
    if hid == "c2":
        dome, out = Bets(), Bets()
        for x in final:
            if x.flags["dome"] == 1.0:
                dome.add(x.over(), x.over_odds, _block(x))
            elif x.flags["dome"] == 0.0:
                out.add(x.over(), x.over_odds, _block(x))
        return {"group1": dome, "group2": out}
    if hid == "d":
        div, non = Bets(), Bets()
        for x in final:
            dog = _underdog(x)
            if dog is None:
                continue
            _side(x, dog, div if x.flags["div"] else non)
        return {"group1": div, "group2": non}
    if hid in ("e1", "e2"):
        hk, ak = ("home_bye_only", "away_bye_only") if hid == "e1" else ("home_short_only", "away_short_only")
        b = Bets()
        for x in final:
            if x.flags[hk]:
                _side(x, "home", b)
            elif x.flags[ak]:
                _side(x, "away", b)
        return {"main": b}
    if hid == "f":
        key, non = Bets(), Bets()
        for x in final:
            dog = _underdog(x)
            if dog is None:
                continue
            _side(x, dog, key if x.flags["key"] else non)
        return {"main": key, "group1": key, "group2": non}
    if hid == "k":
        b = Bets()  # bet AGAINST the streak team (fade); hit = streak team fails to cover
        for x in final:
            hs, as_ = x.flags["home_streak3"], x.flags["away_streak3"]
            if hs and not as_:
                _side(x, "away", b)
            elif as_ and not hs:
                _side(x, "home", b)
        return {"main": b}
    raise ValueError(hid)


DESCRIPTION = {
    "a1": "Home underdogs: home ATS cover", "a2": "Home underdogs in primetime: home ATS cover",
    "j": "Late-season home underdogs: home ATS cover", "b": "Road favorites of 7+: home (underdog) ATS cover",
    "c2": "Over rate, dome minus outdoor", "i": "High totals (≥ 90th pct of prior 3 seasons): under",
    "d": "Underdog cover rate, divisional minus non-divisional", "e1": "Team off a bye vs rested opponent: its ATS cover",
    "e2": "Team on short rest vs non-short opponent: its ATS cover",
    "f": "Underdog ATS cover at spreads of exactly 3 or 7", "k": "Fade a team that covered its last 3: fade side cover",
}


def run_validation(rows: list[MarketRow], n_boot: int = 4000) -> list[dict]:
    results, pvals = [], {}
    for hid in VALIDATION_FAMILY:
        groups = select(rows, hid)
        r = {"id": hid, "description": DESCRIPTION[hid]}
        if hid in ("c2", "d"):
            g1, g2 = groups["group1"], groups["group2"]
            k1, n1, k2, n2 = int(sum(g1.hits)), len(g1.hits), int(sum(g2.hits)), len(g2.hits)
            diff, p, ci = two_prop_z(k1, n1, k2, n2)
            se_null = math.sqrt(0.25 * (1 / n1 + 1 / n2))
            r.update(kind="difference", n1=n1, rate1=k1 / n1, n2=n2, rate2=k2 / n2, estimate=diff, ci=ci, p=p,
                     mde=(Z975 + Z_POWER) * se_null)
            if hid == "d":
                lo, hi = clopper_pearson(k1, n1)
                r["secondary"] = {"div_underdog_rate": k1 / n1, "ci": (lo, hi), "p_vs_50": binom_two_sided_p(k1, n1)}
        else:
            b = groups["main"]
            k, n = int(sum(b.hits)), len(b.hits)
            lo, hi = clopper_pearson(k, n)
            m, rlo, rhi = block_bootstrap_mean_ci(b.returns, b.blocks, n_boot)
            r.update(kind="rate", n=n, pushes=len(b.returns) - n, estimate=k / n, ci=(lo, hi),
                     p=binom_two_sided_p(k, n), mde=mde_rate(n), return_per_unit=m, return_ci=(rlo, rhi))
            if hid == "f":
                g2 = groups["group2"]
                k2, n2 = int(sum(g2.hits)), len(g2.hits)
                diff, p2, ci2 = two_prop_z(k, n, k2, n2)
                r["secondary"] = {"non_key_underdog_rate": k2 / n2, "n_non_key": n2, "key_minus_non_key": diff,
                                  "ci": ci2, "p": p2}
        pvals[hid] = r["p"]
        results.append(r)
    adj = holm(pvals)
    for r in results:
        r["p_holm"] = adj[r["id"]]
        r["supported"] = adj[r["id"]] < 0.05
    return results
