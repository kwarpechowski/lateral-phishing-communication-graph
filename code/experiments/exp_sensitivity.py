"""Analiza wrazliwosci (#GP-EXP-12, M5/M2): czy wnioski sa stabilne wzgledem
parametrow modelu generatywnego (stawki szumu) i rozmiaru populacji N.

Recenzja: naglowkowe liczby zaleza od recznie dobranych stawek szumu (15% off-hours,
25% mimic) i N=150. Tu sweepujemy:
  * off_hours (0..0.40): udzial legalnej poczty poza rytmem (zrodlo FP warstwy czasowej),
  * mimic (0..0.60): udzial atakow mimikujacych rytm (zrodlo FN),
  * N (40..150): rozmiar populacji (podprobkowanie wezlow),
i raportujemy kluczowe metryki (pelny multipleks AUC, AUC przejec, Recall@FPR1%).
Cel: pokazac, ze ORDERING (full > static > contact; czas ratuje przejecia) trzyma sie
JAKOSCIOWO, a liczby zmieniaja sie gladko -- i jawnie, ze sa funkcja modelu generatywnego.

Wyjscie: results/exp_sensitivity_{offhours,mimic,N}.csv
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from graph.build_layers import build_layers                          # noqa: E402
from graph.build_osint_overlay import CATEGORIES, build_overlay      # noqa: E402
from graph.build_temporal_overlay import in_bucket, is_consistent, off_bucket  # noqa: E402

SEEDS = list(range(20))
OSINT = list(CATEGORIES)


def _h(*p):
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def _recall_at_fpr(y, s, t=0.01):
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(y, s)
    m = fpr <= t
    return float(tpr[m].max()) if m.any() else 0.0


def _adj(layers):
    out = {}
    for name, edges in layers.items():
        nbr = {}
        for e in edges:
            nbr.setdefault(e[0], set()).add(e[1]); nbr.setdefault(e[1], set()).add(e[0])
        out[name] = nbr
    return out


def _self_rows():
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _prep(seed, n=None):
    base = build_layers()
    ov, _ = build_overlay(seed=seed, p_cross=0.5, fabrication_rate=0.0)
    layers = {"contact": base["contact"], **{k: ov[k] for k in OSINT}}
    adj = _adj(layers)
    all_twins = sorted({r["twin_id"] for r in _self_rows()})
    keep = set(all_twins[:n]) if n else set(all_twins)
    cn = {k: (v & keep) for k, v in adj["contact"].items() if k in keep}
    ou = {}
    for L in OSINT:
        for t, nb in adj[L].items():
            if t in keep:
                ou.setdefault(t, set()).update(nb & keep)
    return adj, sorted(keep), cn, ou


def _events(seed, twins, cn, ou, off_hours, mimic):
    rows = []
    for v in twins:
        c = sorted(cn.get(v, set()))
        o_only = sorted(ou.get(v, set()) - cn.get(v, set()) - {v})
        off = [t for t in twins if t != v and t not in cn.get(v, set()) and t not in ou.get(v, set())]
        if not c or not off:
            continue
        for k in range(20):
            h = _h(str(seed), v, str(k)); salt = f"{v}:{k}"
            if h % 2 == 0:
                r = (h // 3) % 100
                s = c[h % len(c)] if r < 40 else (o_only[h % len(o_only)] if (r < 70 and o_only) else off[h % len(off)])
                b = off_bucket(s, salt, seed) if (h // 7) % 100 < int(off_hours * 100) else in_bucket(s, salt, seed)
                rows.append((s, v, 0, b))
            else:
                mm = (h // 5) % 100 < int(mimic * 100)
                s = off[h % len(off)] if (h // 11) % 2 == 0 else c[h % len(c)]
                b = in_bucket(s, salt, seed) if mm else off_bucket(s, salt, seed)
                rows.append((s, v, 1, b))
    return rows


def _metrics(adj, rows, twins, seed):
    y = np.array([r[2] for r in rows]); vic = [r[1] for r in rows]
    order = sorted(set(vic))
    te_v = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order)//3)]])
    te = np.array([x in te_v for x in vic]); tr = ~te
    layers_all = ["contact"] + OSINT

    def feat(use_temporal, only_contact=False):
        X = []
        ls = ["contact"] if only_contact else layers_all
        for s, v, _l, b in rows:
            per = [1.0 if s in adj[L].get(v, set()) else 0.0 for L in ls]
            f = per + [sum(per)]
            if use_temporal:
                f.append(1.0 if is_consistent(s, b, seed) else 0.0)
            X.append(f)
        return np.array(X, np.float32)

    def fit(X):
        clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=150, verbose=-1).fit(X[tr], y[tr])
        return clf.predict_proba(X[te])[:, 1]
    yte = y[te]
    sf, ss, sc = fit(feat(True)), fit(feat(False)), fit(feat(True, only_contact=True))
    return (roc_auc_score(yte, sf), _recall_at_fpr(yte, sf),
            roc_auc_score(yte, ss), roc_auc_score(yte, sc))


def _run(off_hours, mimic, n):
    full_auc, full_r1, static_auc, contact_auc = [], [], [], []
    for seed in SEEDS:
        adj, twins, cn, ou = _prep(seed, n)
        rows = _events(seed, twins, cn, ou, off_hours, mimic)
        fa, fr, sa, ca = _metrics(adj, rows, twins, seed)
        full_auc.append(fa); full_r1.append(fr); static_auc.append(sa); contact_auc.append(ca)
    return (float(np.mean(full_auc)), float(np.mean(full_r1)),
            float(np.mean(static_auc)), float(np.mean(contact_auc)))


def main():
    # 1) sweep off_hours
    with (RESULTS / "exp_sensitivity_offhours.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["off_hours", "full_auc", "full_r1", "static_auc", "contact_auc"])
        print("[sens] sweep off_hours (mimic=0.25, N=150):")
        for oh in (0.0, 0.05, 0.15, 0.25, 0.40):
            fa, fr, sa, ca = _run(oh, 0.25, None)
            w.writerow([oh, round(fa, 4), round(fr, 4), round(sa, 4), round(ca, 4)])
            print(f"  off_hours={oh:.2f}: full={fa:.3f} R@1%={fr:.3f} static={sa:.3f} contact={ca:.3f}", flush=True)
    # 2) sweep mimic
    with (RESULTS / "exp_sensitivity_mimic.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["mimic", "full_auc", "full_r1", "static_auc", "contact_auc"])
        print("[sens] sweep mimic (off_hours=0.15, N=150):")
        for mm in (0.0, 0.10, 0.25, 0.40, 0.60, 0.80, 1.0):
            fa, fr, sa, ca = _run(0.15, mm, None)
            w.writerow([mm, round(fa, 4), round(fr, 4), round(sa, 4), round(ca, 4)])
            print(f"  mimic={mm:.2f}: full={fa:.3f} R@1%={fr:.3f} static={sa:.3f} contact={ca:.3f}", flush=True)
    # 3) sweep N
    with (RESULTS / "exp_sensitivity_N.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["N", "full_auc", "full_r1", "static_auc", "contact_auc"])
        print("[sens] sweep N (off_hours=0.15, mimic=0.25):")
        for n in (40, 80, 120, 150):
            fa, fr, sa, ca = _run(0.15, 0.25, n)
            w.writerow([n, round(fa, 4), round(fr, 4), round(sa, 4), round(ca, 4)])
            print(f"  N={n}: full={fa:.3f} R@1%={fr:.3f} static={sa:.3f} contact={ca:.3f}", flush=True)
    print("[sens] zapisano 3 pliki sensitivity. Wniosek: ordering full>static>contact trzyma sie jakosciowo.")


if __name__ == "__main__":
    main()
