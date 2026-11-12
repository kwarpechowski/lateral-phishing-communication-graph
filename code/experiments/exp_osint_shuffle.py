"""Kontrola tasujaca TYLKO warstwy OSINT (recenzja B r1 pkt 3 / r2 pkt 5).

Dotychczasowa kontrola shuffle przemapowywala jednoczesnie nadawce i ofiare, co niszczylo
WSZYSTKIE warstwy (kontakty, OSINT, rytm) — spadek do ~0.51 nie dowodzil therefore, ze sygnal
niosa powiazania OSINT. Tu izolujemy wkad warstw OSINT: permutujemy PRZYNALEZNOSC w kazdej
warstwie OSINT niezaleznie (zachowujac marginesy stopnia warstwy), zostawiajac kontakty i rytm
nietkniete. Jezeli AUC utrzyma sie blisko pelnego multipleksu, sygnal niosa kontakty+rytm;
jezeli spadnie do poziomu contacts+rhythm, wkad spójnosci OSINT jest realny.

Konfiguracje:
  * full       — pelny multiplex + rytm (baseline, porownywalny z exp_ablation "full")
  * osint_shuf — warstwy OSINT permutowane per-warstwa; kontakty i rytm nietkniete
  * all_shuf   — stara kontrola (remap nadawcy+ofiary; niszczy wszystko)
  * no_osint   — tylko kontakty + rytm (bez cech OSINT)

Model zdarzen IDENTYCZNY jak exp_extended (benign known 40/new_osint 30/new_cold 30;
atak bec_off/compromised; 15% off-hours; 25% mimicry). 20 ziaren, LGBM, split po ofiarach.

Wyjscie: results/exp_osint_shuffle.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from exp_extended import REAL_OSINT, _eval, _prepare  # noqa: E402

SEEDS = list(range(20))


def _recall_at_fpr(y, s, t=0.01):
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(y, s)
    m = fpr <= t
    return float(tpr[m].max()) if m.any() else 0.0


def _permute_osint(adj, osint_layers, twins, seed):
    """Permutacja PRZYNALEZNOSCI w kazdej warstwie OSINT niezaleznie: zbior sasiadow wezla t
    w warstwie L trafia do losowego wezla (marginesy warstwy zachowane, kontakty nietkniete)."""
    rng = np.random.default_rng(seed + 4242)
    out = dict(adj)
    for L in osint_layers:
        nodes = list(adj[L].keys())
        perm = list(nodes)
        rng.shuffle(perm)
        m = dict(zip(nodes, perm))
        out[L] = {m[t]: set(nb) for t, nb in adj[L].items()}
    return out


def _boot_ci(vals, n_boot=5000, seed=0):
    """Bootstrap 95% CI sredniej (per-seed resampling; median-of-pairs jak w stats)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return float(v.mean() if len(v) else 0.0), float(v.mean() if len(v) else 0.0)
    bs = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def _score(adj, rows, twins, seed, layer_subset, use_temporal, shuffle_all=False):
    """Jak exp_extended._eval, ale shuffle_all=False => bez remapu; permutacja OSINT idzie
    przez adj (podmieniony przed wywolaniem)."""
    if shuffle_all:
        return _eval(adj, rows, twins, seed, layer_subset, use_temporal, shuffle=True)
    return _eval(adj, rows, twins, seed, layer_subset, use_temporal, shuffle=False)


def main():
    auc_acc = {k: [] for k in ("full", "osint_shuf", "all_shuf", "no_osint")}
    r1_acc = {k: [] for k in auc_acc}
    for seed in SEEDS:
        adj, names, rows, twins = _prepare(seed, p_cross=0.2, real=True)
        osint_layers = [L for L in names if L != "contact"]
        adj_shuf = _permute_osint(adj, osint_layers, twins, seed)
        for tag, a, layers, shuf in (
            ("full", adj, names, False),
            ("osint_shuf", adj_shuf, names, False),
            ("all_shuf", adj, names, True),
            ("no_osint", adj, ["contact"], False),
        ):
            auc, r1 = _score(
                a, rows, twins, seed, layers, use_temporal=True, shuffle_all=shuf
            )
            auc_acc[tag].append(auc)
            r1_acc[tag].append(r1)
        print(
            f"  seed {seed}: "
            + " | ".join(f"{t} auc={np.mean(auc_acc[t]):.3f}" for t in auc_acc),
            flush=True,
        )
    out = RESULTS / "exp_osint_shuffle.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "auc", "auc_ci_lo", "auc_ci_hi",
                     "recall_fpr1", "r1_ci_lo", "r1_ci_hi", "n_seeds"])
        for tag in auc_acc:
            a_lo, a_hi = _boot_ci(auc_acc[tag], seed=1)
            r_lo, r_hi = _boot_ci(r1_acc[tag], seed=2)
            w.writerow(
                [
                    tag,
                    round(float(np.mean(auc_acc[tag])), 4),
                    round(a_lo, 4), round(a_hi, 4),
                    round(float(np.mean(r1_acc[tag])), 4),
                    round(r_lo, 4), round(r_hi, 4),
                    len(auc_acc[tag]),
                ]
            )
    out_ps = RESULTS / "exp_osint_shuffle_perseed.csv"
    with out_ps.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed", "auc", "recall_fpr1"])
        for tag in auc_acc:
            for i, (a, r) in enumerate(zip(auc_acc[tag], r1_acc[tag])):
                w.writerow([tag, i, round(a, 4), round(r, 4)])
    print(f"\n[osint-shuffle] -> {out}")
    for tag in auc_acc:
        a_lo, a_hi = _boot_ci(auc_acc[tag], seed=1)
        print(
            f"  {tag:10s} auc={np.mean(auc_acc[tag]):.3f} [{a_lo:.3f}, {a_hi:.3f}]"
            f"  r1={np.mean(r1_acc[tag]):.3f}"
        )
    for tag in auc_acc:
        print(
            f"  {tag:10s} auc={np.mean(auc_acc[tag]):.3f} r1={np.mean(r1_acc[tag]):.3f}"
        )


if __name__ == "__main__":
    main()
