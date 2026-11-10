"""Krzywa Recall-FPR do niskich progow (M6) + 10 ziaren i test istotnosci (m1) (#GP-EXP-27).

Adresuje uwagi recenzenta: (M6) raportujemy Recall przy FPR in {0.1%,0.5%,1%,5%}, nie tylko 1%;
(m1) 10 ziaren + parowany test istotnosci (Wilcoxon) GNN vs reczny i vs COMPA. Dwa rezimy: blast
(jednobucketowy fanout, exp_cascade._simulate) i stealthy (exp_cascade_stealth, K=2).

Wyjscie: results/exp_cascade_fpr.csv + istotnosc na stdout.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import wilcoxon

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "experiments"))
RESULTS = CODE / "results"

import exp_cascade as XC                                              # noqa: E402
from exp_cascade import (_simulate, _features, _index_traffic,        # noqa: E402
                         _node_features, TemporalGNN)
from exp_cascade_stealth import _simulate_stealth                     # noqa: E402
from exp_cascade_sweep import _compa_features, _per_bucket            # noqa: E402

SEEDS = list(range(10))
EPOCHS = 100
FPRS = [0.001, 0.005, 0.01, 0.05]
MODELS = ["tab_ctx", "compa", "gnn_temporal"]


def _recalls(y, s, fprs):
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return {f: 0.0 for f in fprs}
    fpr, tpr, _ = roc_curve(y, s)
    out = {}
    for f in fprs:
        m = fpr <= f
        out[f] = float(tpr[m].max()) if m.any() else 0.0
    return out


def _one(seed, regime):
    torch.manual_seed(seed)
    if regime == "blast":
        twins, nbr, events = _simulate(seed)
    else:
        twins, nbr, events = _simulate_stealth(seed, 2)
    recv_b, sent_b = _index_traffic(events)
    X1, Xc, _y = _features(events, nbr, recv_b, sent_b, seed)
    Xco = _compa_features(events)
    idx = {t: i for i, t in enumerate(twins)}; n = len(twins)
    y = np.array([e[3] for e in events], dtype=np.float32)
    vics = sorted({e[1] for e in events})
    rng = np.random.default_rng(seed)
    te_v = set(np.array(vics)[rng.permutation(len(vics))[:max(1, len(vics) // 3)]])
    te = np.array([e[1] in te_v for e in events]); tr = ~te
    scores = {}
    for tag, X in (("tab_ctx", Xc), ("compa", Xco)):
        c = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(X[tr], y[tr])
        scores[tag] = c.predict_proba(X[te])[:, 1]
    Xn = _node_features(events, nbr, idx, n, recv_b, sent_b, tr=tr)
    s_idx = torch.tensor([idx[e[0]] for e in events]); v_idx = torch.tensor([idx[e[1]] for e in events])
    ef = torch.tensor(X1, dtype=torch.float32); yt = torch.tensor(y); tr_t = torch.tensor(tr)
    pb = _per_bucket(events, idx)
    m = TemporalGNN(Xn.shape[1], X1.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4); lf = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        loss = lf(m(Xn, pb, n, ef, s_idx, v_idx)[tr_t], yt[tr_t]); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        scores["gnn_temporal"] = torch.sigmoid(m(Xn, pb, n, ef, s_idx, v_idx))[torch.tensor(te)].numpy()
    yte = y[te]
    res = {}
    for tag in MODELS:
        rec = _recalls(yte, scores[tag], FPRS)
        res[tag] = {"auc": roc_auc_score(yte, scores[tag]), "rec": rec}
    return res


def main():
    rows = []
    for regime in ["blast", "stealthy"]:
        agg = {tag: {"auc": [], **{f: [] for f in FPRS}} for tag in MODELS}
        for seed in SEEDS:
            r = _one(seed, regime)
            for tag in MODELS:
                agg[tag]["auc"].append(r[tag]["auc"])
                for f in FPRS:
                    agg[tag][f].append(r[tag]["rec"][f])
            print(f"  [{regime}] seed={seed} done", flush=True)
        print(f"\n=== {regime} (10 ziaren) ===")
        for tag in MODELS:
            line = " ".join(f"R@{f}={np.mean(agg[tag][f]):.3f}" for f in FPRS)
            print(f"  {tag:13s} AUC={np.mean(agg[tag]['auc']):.3f}  {line}")
            for f in FPRS:
                rows.append([regime, tag, f, round(float(np.mean(agg[tag][f])), 4),
                             round(float(np.mean(agg[tag]['auc'])), 4)])
        # istotnosc: GNN vs reczny i vs COMPA (parowany Wilcoxon na AUC, 10 ziaren)
        for opp in ["tab_ctx", "compa"]:
            try:
                stat, p = wilcoxon(agg["gnn_temporal"]["auc"], agg[opp]["auc"])
                d = np.mean(agg["gnn_temporal"]["auc"]) - np.mean(agg[opp]["auc"])
                print(f"  Wilcoxon GNN vs {opp}: dAUC={d:+.3f}, p={p:.4f}")
            except Exception as e:
                print(f"  Wilcoxon GNN vs {opp}: {e}")
    out = RESULTS / "exp_cascade_fpr.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["regime", "model", "fpr", "recall", "auc"])
        w.writerows(rows)
    print(f"\n[fpr] -> {out}")


if __name__ == "__main__":
    main()
