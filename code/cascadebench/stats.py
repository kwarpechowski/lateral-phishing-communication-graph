"""cascadebench.stats — rygor statystyczny dla raportowanych liczb.

Adresuje recenzje statystyczna: bootstrap-CI zamiast pojedynczych srednich, kwantyl t (nie 1.96)
przy malym n, korekta Holma na wielokrotne porownania, rozmiar efektu Cliff's delta, test parowany.
Deterministyczny (jawne ziarno) — bez Math.random/Date.now (spojnosc z reszta pakietu).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Dict
import numpy as np


def mean_ci(values: Sequence[float], alpha: float = 0.05, method: str = "bootstrap",
            n_boot: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    """Zwraca (srednia, dol_CI, gora_CI). method='bootstrap' (percentylowy) lub 't' (kwantyl Studenta).
    Bootstrap jest odporny na maly n i nie zaklada normalnosci; 't' jako szybka alternatywa."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, m, m
    if method == "t":
        from scipy.stats import t
        se = x.std(ddof=1) / np.sqrt(n)
        q = t.ppf(1 - alpha / 2, df=n - 1)
        return m, m - q * se, m + q * se
    rng = np.random.default_rng(seed)
    boot = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return m, float(lo), float(hi)


def fmt_ci(values: Sequence[float], prec: int = 3, **kw) -> str:
    """Format 'srednia [dol, gora]' do raportu/LaTeX."""
    m, lo, hi = mean_ci(values, **kw)
    return f"{m:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> Tuple[float, str]:
    """Cliff's delta — nieparametryczny rozmiar efektu (a vs b). Progi: |d|<0.147 znikomy,
    <0.33 maly, <0.474 sredni, inaczej duzy (Romano i in.)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return 0.0, "n/d"
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    d = (gt - lt) / (len(a) * len(b))
    ad = abs(d)
    mag = "znikomy" if ad < 0.147 else "maly" if ad < 0.33 else "sredni" if ad < 0.474 else "duzy"
    return float(d), mag


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Parowany test Wilcoxona (te same ziarna). Zwraca (statystyka, p). UWAGA: przy n<6 dwustronne p
    ma PODLOGE (>0.05 mozliwe nieosiagalne) — raportowac to wprost."""
    from scipy.stats import wilcoxon
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.allclose(a, b):
        return 0.0, 1.0
    try:
        stat, p = wilcoxon(a, b)
        return float(stat), float(p)
    except ValueError:
        return float("nan"), 1.0


def holm(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, Tuple[float, bool]]:
    """Korekta Holma-Bonferroniego na wielokrotne porownania. Zwraca nazwa -> (p_skorygowane, istotne)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Tuple[float, bool]] = {}
    prev = 0.0
    for i, (name, p) in enumerate(items):
        p_adj = min(1.0, max(prev, (m - i) * p))   # monotonicznosc
        prev = p_adj
        out[name] = (p_adj, p_adj < alpha)
    return out


def min_two_sided_wilcoxon_p(n: int) -> float:
    """Najmniejsze osiagalne dwustronne p testu znakowanych rang dla n par (podloga mocy).
    Np. n=5 -> 0.0625 (>0.05 NIEOSIAGALNE). Sluzy do uczciwego raportu mocy."""
    return 2.0 / (2 ** n) * 1.0 if n <= 0 else 2.0 / (2 ** n)
