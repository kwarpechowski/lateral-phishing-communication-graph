# -*- coding: utf-8 -*-
"""B2 (recenzja wewnetrzna r2): rozdzielenie bitu tc na DWIE cechy i ablacja.

Dotychczasowa pojedyncza cecha czasowa tc ('nadawca w swoim rytmie') bywa mylona z
'sender known' (nadawca jest kontaktem ofiary). Rozdzielamy jawnie:
  * known_sender — nadawca jest w warstwie kontaktow ofiary (feature warstwy contact),
  * in_rhythm    — nadawce pisze w swoich typowych godzinach (bit tc),
i liczymy ablacje: kazda osobno, obie razem, oraz punkty odniesienia (pelny multipleks
bez tc / pelny z tc). 20 ziaren, realne warstwy OSINT, bootstrap-CI, per-seed dump.

Wyjscie: results/exp_tc_ablation.csv + _perseed.csv
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
sys.path.insert(0, str(CODE / "experiments"))
RESULTS = CODE / "results"

sys.argv = [sys.argv[0]]

from exp_extended import REAL_OSINT, _prepare  # noqa: E402

SEEDS = list(range(20))
N_BOOT = 5000
FPR = 0.01


def _boot_ci(vals, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return 0.0, 0.0
    bs = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def _recall_at_fpr(y, s, t=FPR):
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(y, s)
    m = fpr <= t
    return float(tpr[m].max()) if m.any() else 0.0


def _score(adj, rows, twins, seed, layer_subset, use_temporal):
    """Jak exp_extended._eval (split po ofiarach, LGBM).

    R4-B3: in_rhythm = NaN (brakujaca) dla nadawcow BEZ HISTORII w danych
    organizacji (nadawcy, ktorzy nie wystepuja jako nadawcy zadnego zdarzenia
    benign — organizacja nie ma z czego estymowac rytmu). LGBM obsluguje NaN
    natywnie; wczesniej tacy nadawcy dostawali hard-coded 0 (ataki byly z
    definicji off-rhythm), co bylo artefaktem dostepnosci danych."""
    y = np.array([r[2] for r in rows])
    vic = [r[1] for r in rows]
    order = sorted(set(vic))
    te_v = set(
        np.array(order)[
            np.random.default_rng(seed).permutation(len(order))[
                : max(1, len(order) // 3)
            ]
        ]
    )
    te = np.array([x in te_v for x in vic])
    tr = ~te
    from graph.build_temporal_overlay import is_consistent

    # R4-B3: nadawcy z historia = nadawcy zdarzen benign obserwowanych przez organizacje
    seen_senders = {r[0] for r in rows if r[2] == 0}
    X = []
    for s, v, _l, _a, b, fab in rows:
        per = [
            1.0 if (s in adj[L].get(v, set()) or L == fab) else 0.0
            for L in layer_subset
        ]
        feat = per + [sum(per)]
        if use_temporal:
            if s in seen_senders:
                feat.append(1.0 if is_consistent(s, b, seed) else 0.0)
            else:
                feat.append(float("nan"))  # brak historii -> cecha brakujaca
        X.append(feat)
    X = np.array(X, np.float32)
    clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(
        X[tr], y[tr]
    )
    return clf.predict_proba(X[te])[:, 1], y[te]


def main():
    configs = [
        ("known_sender (kontakty)", ["contact"], False),
        ("in_rhythm (tc)", [], True),
        ("known_sender + in_rhythm", ["contact"], True),
        ("multiplex bez tc", ["contact"] + REAL_OSINT, False),
        ("multiplex + tc (pelny)", ["contact"] + REAL_OSINT, True),
    ]
    acc = {name: {"auc": [], "r1": []} for name, _ls, _ut in configs}
    perseed = []
    for seed in SEEDS:
        adj, names, rows, twins = _prepare(seed, 0.5, 0.0, real=True)
        for name, ls, ut in configs:
            s, yte = _score(adj, rows, twins, seed, ls, ut)
            a = roc_auc_score(yte, s)
            r = _recall_at_fpr(yte, s)
            acc[name]["auc"].append(a)
            acc[name]["r1"].append(r)
            perseed.append([name, seed, round(a, 4), round(r, 4)])
        print(f"  seed {seed} done", flush=True)

    out = RESULTS / "exp_tc_ablation.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "config",
                "auc",
                "auc_ci_lo",
                "auc_ci_hi",
                "recall_fpr1",
                "r1_ci_lo",
                "r1_ci_hi",
                "n_seeds",
            ]
        )
        for name, _ls, _ut in configs:
            a, r = acc[name]["auc"], acc[name]["r1"]
            alo, ahi = _boot_ci(a, seed=1)
            rlo, rhi = _boot_ci(r, seed=2)
            w.writerow(
                [
                    name,
                    round(float(np.mean(a)), 4),
                    round(alo, 4),
                    round(ahi, 4),
                    round(float(np.mean(r)), 4),
                    round(rlo, 4),
                    round(rhi, 4),
                    len(a),
                ]
            )
    out_ps = RESULTS / "exp_tc_ablation_perseed.csv"
    with out_ps.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed", "auc", "recall_fpr1"])
        w.writerows(perseed)
    print(f"\n[tc-ablation] -> {out}")
    for name, _ls, _ut in configs:
        print(
            f"  {name:28s} auc={np.mean(acc[name]['auc']):.3f} "
            f"r1={np.mean(acc[name]['r1']):.3f}"
        )


if __name__ == "__main__":
    main()
