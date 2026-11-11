"""Jawny operator SPOJNOSCI miedzywarstwowej vs konkatenacja (#GP-EXP-11, M3 / H8c).

Recenzja: twierdzimy "spojnosc > konkatenacja", ale glowny model to konkatenacja cech
per-warstwa (drzewa ucza sie koniunkcji za darmo, wiec roznica jest ukryta). Tu testujemy
to wprost: definiujemy JAWNE score'y (nie)spojnosci miedzywarstwowej --

  cov_osint        = liczba warstw OSINT laczacych nadawce z ofiara
  compromised_incons = kontakt=1 AND poza rytmem  (znany nadawca, ale niespojny czasowo)
  newsender_consist  = kontakt=0 AND cov_osint>0  (nowy nadawca, lecz spojny przez OSINT)
  fully_off          = brak jakiejkolwiek warstwy AND poza rytmem

-- i porownujemy je z surowa konkatenacja pod DWOMA klasyfikatorami:
  * LogisticRegression: liniowy, NIE uczy sie koniunkcji -> jawny operator powinien wygrac.
  * LightGBM: drzewa ucza sie koniunkcji -> roznica zanika (operator jest implicit).

Pokazuje, ze sygnalem jest *cross-layer inconsistency*, a nie generyczna fuzja -- co odroznia
nas od multi-view fusion (obrona nowosci). Wyjscie: results/exp_consistency.csv
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from graph.build_layers import build_layers                          # noqa: E402
from graph.build_osint_overlay import CATEGORIES, build_overlay      # noqa: E402
from graph.build_temporal_overlay import in_bucket, is_consistent, off_bucket  # noqa: E402

SEEDS = list(range(20))
# Warstwy OSINT z REALNYCH pol blizniaka (c3-c8), nie syntetyczne spolecznosci (spojnie z exp_fusion).
OSINT = ["osint_conference", "osint_certification", "osint_skill", "osint_routine",
         "osint_event", "osint_pretext", "osint_platform", "interest_sim"]


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


def _prep(seed):
    base = build_layers()
    layers = {"contact": base["contact"], **{k: base[k] for k in OSINT}}
    adj = _adj(layers)
    cn = adj["contact"]
    ou = {}
    for L in OSINT:
        for t, nb in adj[L].items():
            ou.setdefault(t, set()).update(nb)
    twins = sorted(set(cn) | set(ou) | {r["twin_id"] for r in _self_rows()})
    return adj, twins, cn, ou


def _events(seed, twins, cn, ou):
    rows = []
    for v in twins:
        c = sorted(cn.get(v, set()))
        o_only = sorted(ou.get(v, set()) - cn.get(v, set()) - {v})
        off = [t for t in twins if t != v and t not in cn.get(v, set()) and t not in ou.get(v, set())]
        if not c or not off:
            continue
        for k in range(24):
            h = _h(str(seed), v, str(k)); salt = f"{v}:{k}"
            if h % 2 == 0:
                r = (h // 3) % 100
                s = c[h % len(c)] if r < 40 else (o_only[h % len(o_only)] if (r < 70 and o_only) else off[h % len(off)])
                b = off_bucket(s, salt, seed) if (h // 7) % 100 < 15 else in_bucket(s, salt, seed)
                rows.append((s, v, 0, b))
            else:
                mimic = (h // 5) % 100 < 25
                s = off[h % len(off)] if (h // 11) % 2 == 0 else c[h % len(c)]
                b = in_bucket(s, salt, seed) if mimic else off_bucket(s, salt, seed)
                rows.append((s, v, 1, b))
    return rows


def _features(adj, rows, seed):
    """Zwraca (X_concat, X_consistency)."""
    Xc, Xk = [], []
    for s, v, _l, b in rows:
        contact = 1.0 if s in adj["contact"].get(v, set()) else 0.0
        osint = [1.0 if s in adj[L].get(v, set()) else 0.0 for L in OSINT]
        tcon = 1.0 if is_consistent(s, b, seed) else 0.0
        cov = sum(osint)
        Xc.append([contact] + osint + [tcon])                     # surowa konkatenacja
        Xk.append([                                               # JAWNE score'y spojnosci
            cov,                                                  # pokrycie OSINT
            contact * (1.0 - tcon),                               # niespojnosc: znany ale poza rytmem
            (1.0 - contact) * (1.0 if cov > 0 else 0.0),          # spojnosc: nowy, ale na OSINT
            1.0 if (contact == 0 and cov == 0 and tcon == 0) else 0.0,  # calkiem off
        ])
    return np.array(Xc, np.float32), np.array(Xk, np.float32)


def run(seed):
    adj, twins, cn, ou = _prep(seed)
    rows = _events(seed, twins, cn, ou)
    Xc, Xk = _features(adj, rows, seed)
    y = np.array([r[2] for r in rows]); vic = [r[1] for r in rows]
    order = sorted(set(vic))
    te_v = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order)//3)]])
    te = np.array([x in te_v for x in vic]); tr = ~te

    def lr(X):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        return clf.predict_proba(sc.transform(X[te]))[:, 1]

    def gb(X):
        clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(X[tr], y[tr])
        return clf.predict_proba(X[te])[:, 1]
    yte = y[te]
    out = {}
    for tag, sc in [("LR_concat", lr(Xc)), ("LR_consist", lr(Xk)),
                    ("GB_concat", gb(Xc)), ("GB_consist", gb(Xk))]:
        out[f"{tag}_auc"] = roc_auc_score(yte, sc)
        out[f"{tag}_r1"] = _recall_at_fpr(yte, sc)
    return out


def main():
    acc = {}
    for seed in SEEDS:
        for k, v in run(seed).items():
            acc.setdefault(k, []).append(v)
    agg = {k: float(np.mean(v)) for k, v in acc.items()}
    out = RESULTS / "exp_consistency.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model", "auc", "recall_fpr1"])
        for tag in ("LR_concat", "LR_consist", "GB_concat", "GB_consist"):
            w.writerow([tag, round(agg[f"{tag}_auc"], 4), round(agg[f"{tag}_r1"], 4)])
    print("[consistency] jawny operator spojnosci vs konkatenacja:")
    print(f"  {'model':12s} {'AUC':>7s} {'R@1%':>7s}")
    for tag in ("LR_concat", "LR_consist", "GB_concat", "GB_consist"):
        print(f"  {tag:12s} {agg[tag+'_auc']:7.3f} {agg[tag+'_r1']:7.3f}")
    print("Oczekiwane: LR_consist >> LR_concat (operator pomaga modelowi liniowemu); "
          "GB_concat ~ GB_consist (drzewa ucza sie koniunkcji).")
    print(f"[consistency] wrote {out}")


if __name__ == "__main__":
    main()
