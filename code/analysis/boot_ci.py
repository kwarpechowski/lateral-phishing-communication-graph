# -*- coding: utf-8 -*-
"""A6: bootstrap 95% CI po seedach z plikow *_perseed.csv (recenzja wewnetrzna r2).

Dla kazdego pliku results/*_perseed.csv: grupuje wiersze po wszystkich kolumnach
oprocz 'seed' i kolumny-numeryka (pierwsza kolumna o nazwie z {auc, value, r1,
recall_fpr1, sens}), liczy srednia + bootstrap-CI95 (5000 resampli, seed 0).

Wyjscie: results/ci/<nazwa>_ci.csv  (kolumny: grupa..., mean, ci_lo, ci_hi, n_seeds)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
RESULTS = CODE / "results"
OUT = RESULTS / "ci"
N_BOOT = 5000
METRIC_CANDIDATES = ["auc", "value", "r1", "recall_fpr1", "sens", "recall_fpr1_mean"]


def _boot_ci(vals, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else 0.0), (
            float(v.mean()) if len(v) else 0.0
        )
    bs = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def process(path: Path) -> str | None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return None
    cols = list(rows[0].keys())
    metric = next((c for c in METRIC_CANDIDATES if c in cols), None)
    if metric is None:
        return f"  [skip] {path.name} (brak kolumny metryki z {METRIC_CANDIDATES})"
    keys = [c for c in cols if c not in (metric, "seed")]
    groups = {}
    for r in rows:
        k = tuple(r[c] for c in keys)
        try:
            val = float(r[metric])
        except (TypeError, ValueError):
            continue
        groups.setdefault(k, []).append(val)
    out = OUT / (path.stem.replace("_perseed", "") + "_ci.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys + ["mean", "ci_lo", "ci_hi", "n_seeds"])
        for k in sorted(groups):
            v = groups[k]
            lo, hi = _boot_ci(v, seed=1)
            w.writerow(
                list(k)
                + [round(float(np.mean(v)), 4), round(lo, 4), round(hi, 4), len(v)]
            )
    return f"  [ok] {out.name} ({len(groups)} grup)"


def main():
    OUT.mkdir(exist_ok=True)
    for p in sorted(RESULTS.glob("*_perseed.csv")):
        msg = process(p)
        if msg:
            print(msg)


if __name__ == "__main__":
    main()
