"""Fuzja TRESCI z multipleksem: kiedy tresc pomaga, a kiedy multipleks (#GP-EXP-7).

Adresuje granice multipleksu (impersonacja spoza grafu, ktorej nie lapal przy FPR=1%):
czy dolaczenie sygnalu tresci ja zamyka? Pokazuje komplementarnosc dwoch rezimow:
  * TELL-tale: atak niesie slad tresciowy (realny spear-phish) -> tresc separuje, multipleks
    redundantny.
  * MIMIKRA (BEC): atak mimikuje legalna tresc (tekst benign, zlosliwy jest nadawca) -> tresc
    slepa (~0.5), multipleks/czas niosa sygnal.
Porownanie: tylko-tresc / tylko-multipleks / fuzja, w obu rezimach. LightGBM, 5 ziaren,
podzial po ofiarach, AUC + Recall@FPR=1%. Wyjscie: results/exp_content_fusion.csv
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from graph.build_layers import build_layers                          # noqa: E402
from graph.build_osint_overlay import build_overlay                  # noqa: E402
from graph.build_temporal_overlay import in_bucket, is_consistent, off_bucket  # noqa: E402

SEEDS = list(range(20))
OSINT = ["osint_mailing_list", "osint_oss_project", "osint_affiliation"]


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
            nbr.setdefault(e[0], set()).add(e[1])
            nbr.setdefault(e[1], set()).add(e[0])
        out[name] = nbr
    return out


def _load_corpus():
    ben = pd.read_csv(RESULTS / "benign_corpus.csv")
    atk = pd.read_csv(RESULTS / "generated_corpus.csv")
    bt = {}
    for t, g in ben.groupby("profile_id"):
        bt[t] = list(g["text"].astype(str))
    at = {}
    for t, g in atk.groupby("profile_id"):
        at[t] = list(g["text"].astype(str))
    allben = list(ben["text"].astype(str))
    return bt, at, allben


def run(seed, regime):
    base = build_layers()
    ov, _ = build_overlay(seed=seed, p_cross=0.5, fabrication_rate=0.0)
    layers = {"contact": base["contact"], **ov}
    adj = _adj(layers)
    contact_nbr = adj["contact"]
    osint_union = {}
    for L in OSINT:
        for t, nb in adj[L].items():
            osint_union.setdefault(t, set()).update(nb)
    bt, at, allben = _load_corpus()
    twins = sorted(set(contact_nbr) & set(bt))             # blizniaki z tekstem benign
    rows = []   # (text, label, sender, victim, bucket)
    for v in twins:
        c = sorted(contact_nbr.get(v, set()))
        o_only = sorted(osint_union.get(v, set()) - contact_nbr.get(v, set()) - {v})
        off = [t for t in twins if t != v and t not in contact_nbr.get(v, set())
               and t not in osint_union.get(v, set())]
        if not c or not off or v not in bt:
            continue
        for k in range(16):
            h = _h(str(seed), v, str(k)); salt = f"{v}:{k}"
            ti = _h("txt", str(seed), v, str(k)) % len(bt[v])   # indeks tekstu NIEZALEzny od etykiety
            btxt = bt[v][ti]                               # wlasny benign ofiary (jednolity rozklad)
            if h % 2 == 0:                                  # benign (known / new_osint / new_cold)
                r = (h // 3) % 100
                if r < 40:
                    s = c[h % len(c)]
                elif r < 70 and o_only:
                    s = o_only[h % len(o_only)]
                else:
                    s = off[h % len(off)]
                b = off_bucket(s, salt, seed) if (h // 7) % 100 < 15 else in_bucket(s, salt, seed)
                rows.append((btxt, 0, s, v, b))
            else:                                          # atak (bec_off / compromised)
                mimic = (h // 5) % 100 < 25
                s = off[h % len(off)] if (h // 11) % 2 == 0 else c[h % len(c)]
                b = in_bucket(s, salt, seed) if mimic else off_bucket(s, salt, seed)
                if regime == "tell" and v in at and at[v]:
                    txt = at[v][h % len(at[v])]            # realny spear-phish (slad tresciowy)
                else:
                    txt = btxt                             # mimikra: ten SAM rozklad co benign -> tresc slepa
                rows.append((txt, 1, s, v, b))
    texts = [r[0] for r in rows]; y = np.array([r[1] for r in rows])
    vic = [r[3] for r in rows]
    order = sorted(set(vic))
    te_v = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order)//3)]])
    te = np.array([x in te_v for x in vic]); tr = ~te

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000).fit([t for t, m in zip(texts, tr) if m])
    Xc_tr = vec.transform([t for t, m in zip(texts, tr) if m])
    Xc_te = vec.transform([t for t, m in zip(texts, te) if m])
    mux = np.array([[1.0 if s in adj[L].get(v, set()) else 0.0 for L in ["contact"] + OSINT]
                    + [1.0 if is_consistent(s, b, seed) else 0.0]
                    for _t, _l, s, v, b in rows], dtype=np.float32)
    ytr, yte = y[tr], y[te]

    def fit(Xtr, Xte):
        clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    s_c = fit(Xc_tr, Xc_te)
    s_m = fit(csr_matrix(mux[tr]), csr_matrix(mux[te]))
    s_f = fit(hstack([Xc_tr, csr_matrix(mux[tr])]).tocsr(), hstack([Xc_te, csr_matrix(mux[te])]).tocsr())
    return {
        "content_auc": roc_auc_score(yte, s_c), "content_r1": _recall_at_fpr(yte, s_c),
        "mux_auc": roc_auc_score(yte, s_m), "mux_r1": _recall_at_fpr(yte, s_m),
        "fusion_auc": roc_auc_score(yte, s_f), "fusion_r1": _recall_at_fpr(yte, s_f),
    }


def main():
    out = RESULTS / "exp_content_fusion.csv"
    rowsout = []
    print(f"{'rezim':8s} | {'content AUC/R@1%':>18s} | {'mux AUC/R@1%':>16s} | {'fuzja AUC/R@1%':>16s}")
    for regime in ("tell", "mimicry"):
        acc = {}
        for seed in SEEDS:
            for k, v in run(seed, regime).items():
                acc.setdefault(k, []).append(v)
        agg = {k: float(np.mean(v)) for k, v in acc.items()}
        rowsout.append({"regime": regime, **{k: round(v, 3) for k, v in agg.items()}})
        print(f"{regime:8s} | {agg['content_auc']:.3f} / {agg['content_r1']:.3f}      | "
              f"{agg['mux_auc']:.3f} / {agg['mux_r1']:.3f}    | {agg['fusion_auc']:.3f} / {agg['fusion_r1']:.3f}")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rowsout[0])); w.writeheader(); w.writerows(rowsout)
    print(f"[content-fusion] wrote {out}")


if __name__ == "__main__":
    main()
