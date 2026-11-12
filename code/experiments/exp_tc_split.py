# -*- coding: utf-8 -*-
"""TC-SPLIT (paper B, fix po recenzji zewnetrznej): GDZIE multiplex wnosi sygnal?

Globalne AUC nie mowi, ktory podzbior zdarzen naprawde korzysta na multipleksie.
Trencujemy raz per seed (pelny multiplex + rytm vs tylko kontakty + rytm), a metryki
(AUC, Recall@FPR1%) liczymy OSOBNIE na trzech przecinajacych sie podzbiorach testowych:

  * known-sender — nadawca jest kontaktem ofiary (benign 'known' + atak 'compromised');
    rozroznienie niosa rytm/wzor czasowy; hipoteza: multiplex ~= contact-only
  * in-rhythm    — nadawca pisze w swoich typowych godzinach (is_consistent); najtrudniejszy
    plaster (atak mimicry); hipoteza: to tu multipleks buduje przewage
  * new-sender   — nadawca NIE jest kontaktem (benign 'new_osint'/'new_cold' + atak
    'bec_off'); hipoteza: warstwy OSINT ratuja legitne first-contacty przed FP

Kontrola: shuffle OSINT (permutacja przynaleznosci per warstwa) — spadek na plastrze
in-rhythm potwierdza przyczynowosc wkadu OSINT (a nie sam fakt dodatkowych cech).

Model zdarzen IDENTYCZNY jak exp_extended (20 ziaren, realne warstwy, split po ofiarach,
LGBM). Wyjscie: results/exp_tc_split.csv + _perseed.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "experiments"))
RESULTS = CODE / "results"

SEEDS = list(range(20))
N_BOOT = 5000

sys.argv = [sys.argv[0]]  # wyczysc argv przed importem exp_extended

from exp_extended import REAL_OSINT, _prepare, _self_rows  # noqa: E402
from exp_osint_shuffle import _permute_osint  # noqa: E402
from graph.build_temporal_overlay import is_consistent  # noqa: E402


def _boot_ci(vals, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else 0.0), (
            float(v.mean()) if len(v) else 0.0
        )
    bs = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def _recall_at_fpr(y, s, t=0.01):
    if y.sum() == 0 or (1 - y).sum() == 0:
        return 0.0
    order = np.argsort(-s)
    y = np.asarray(y)[order]
    s = np.asarray(s)[order]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_neg == 0:
        return 0.0
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    ok = np.where(fp <= t * n_neg)[0]
    return float(tp[ok[-1]] / n_pos) if len(ok) else 0.0


def _slices(rows, contact_nbr, seed):
    """Maski podzbiorow: (nazwa, maska). known/in_rhythm/new sa niespojne w parach ale
    moga sie przecinac (known AND in_rhythm) — to celowe: plaster 'in-rhythm' zawiera tez
    znane kontakty piszace w rytmie."""
    known, inr, new = [], [], []
    for s, v, _l, st, b, _f in rows:
        is_known = (
            (s in contact_nbr.get(v, set())) or (st == "known") or (st == "compromised")
        )
        known.append(is_known)
        inr.append(bool(is_consistent(s, b, seed)))
        new.append(not is_known)
    return {
        "known-sender": np.array(known),
        "in-rhythm": np.array(inr),
        "new-sender": np.array(new),
    }


def run_config(seed, adj, rows, twins, layer_subset, use_temporal=True):
    """Trenuj na calym train-split; zwroc score + y + slice-maski na test-split."""
    wrong = None  # bez shuffle tu
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
    X = []
    for s, v, _l, _a, b, fab in rows:
        per = [
            1.0 if (s in adj[L].get(v, set()) or L == fab) else 0.0
            for L in layer_subset
        ]
        feat = per + [sum(per)]
        if use_temporal:
            feat.append(1.0 if is_consistent(s, b, seed) else 0.0)
        X.append(feat)
    X = np.array(X, np.float32)
    clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(
        X[tr], y[tr]
    )
    sc = clf.predict_proba(X[te])[:, 1]
    contact_nbr = adj["contact"]
    sl = {k: v[te] for k, v in _slices(rows, contact_nbr, seed).items()}
    return y[te], sc, sl


def main():
    configs = {
        "contact+rytm": None,  # pelny multiplex — warstwy ustawiane per seed
        "multiplex+rytm": None,
        "multiplex+rytm (osint-shuf)": None,
    }
    acc = {
        c: {
            sl: {"auc": [], "r1": []}
            for sl in ("all", "known-sender", "in-rhythm", "new-sender")
        }
        for c in configs
    }
    perseed = []
    for seed in SEEDS:
        adj, names, rows, twins = _prepare(seed, p_cross=0.2, real=True)
        osint_layers = [L for L in names if L != "contact"]
        adj_shuf = _permute_osint(adj, osint_layers, twins, seed)
        variants = {
            "contact+rytm": (adj, ["contact"]),
            "multiplex+rytm": (adj, names),
            "multiplex+rytm (osint-shuf)": (adj_shuf, names),
        }
        for cname, (a, subset) in variants.items():
            yte, sc, sl = run_config(seed, a, rows, twins, subset, use_temporal=True)
            subsets = {"all": np.ones(len(yte), bool), **sl}
            for sname, m in subsets.items():
                if (yte[m] == 0).all() or (yte[m] == 1).all():
                    continue
                a_ = roc_auc_score(yte[m], sc[m])
                r_ = _recall_at_fpr(yte[m], sc[m])
                acc[cname][sname]["auc"].append(a_)
                acc[cname][sname]["r1"].append(r_)
                perseed.append([cname, sname, seed, round(a_, 4), round(r_, 4)])
        print(f"  seed {seed} done", flush=True)

    out = RESULTS / "exp_tc_split.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "config",
                "slice",
                "auc",
                "auc_ci_lo",
                "auc_ci_hi",
                "recall_fpr1",
                "r1_ci_lo",
                "r1_ci_hi",
                "n_seeds",
            ]
        )
        for cname in configs:
            for sname in ("all", "known-sender", "in-rhythm", "new-sender"):
                a = acc[cname][sname]["auc"]
                r = acc[cname][sname]["r1"]
                if not a:
                    continue
                a_lo, a_hi = _boot_ci(a, seed=1)
                r_lo, r_hi = _boot_ci(r, seed=2)
                w.writerow(
                    [
                        cname,
                        sname,
                        round(float(np.mean(a)), 4),
                        round(a_lo, 4),
                        round(a_hi, 4),
                        round(float(np.mean(r)), 4),
                        round(r_lo, 4),
                        round(r_hi, 4),
                        len(a),
                    ]
                )
    out_ps = RESULTS / "exp_tc_split_perseed.csv"
    with out_ps.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "slice", "seed", "auc", "recall_fpr1"])
        w.writerows(perseed)
    print(f"\n[tc-split] -> {out}")
    for cname in configs:
        for sname in ("all", "known-sender", "in-rhythm", "new-sender"):
            a = acc[cname][sname]["auc"]
            if a:
                print(f"  {cname:28s} {sname:12s} auc={np.mean(a):.3f} (n={len(a)})")


if __name__ == "__main__":
    main()
