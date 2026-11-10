"""Kotwica na WIELU realnych topologiach komunikacji (#GP-EXP-26) — adresuje M2 (tylko Enron).

Powtarza matched-design eksperyment kaskady na publicznych, anonimizowanych grafach komunikacji
ze SNAP (zero badan na ludziach): email-Eu-core (realny e-mail instytucji) i CollegeMsg (realne
wiadomosci). Realna TOPOLOGIA + wstrzykniete kaskady + matched benign (jak kotwica Enron).
Detektory: reczny kontekst, COMPA (wolumenowy), temporalny GNN, kontrola shuffle.

Wyjscie: results/exp_cascade_realgraphs.csv
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "experiments"))
RESULTS = CODE / "results"
GRAPHS = CODE / "data" / "realgraphs"

from exp_cascade import (T_BUCKETS, _features, _index_traffic,        # noqa: E402
                         _node_features, _recall_at_fpr, TemporalGNN)
from exp_cascade_sweep import _compa_features, _per_bucket            # noqa: E402

SEEDS = list(range(3))
EPOCHS = 90
CORE_CAP = 2000
MIN_DEG = 2
MAX_HOPS = 3
MAX_FANOUT = 10
P_INFECT = 0.6
DATASETS = [("email-Eu-core", "eu.txt"), ("CollegeMsg", "college.txt")]


def load_graph(fname):
    nbr = defaultdict(set)
    with (GRAPHS / fname).open() as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            u, v = p[0], p[1]
            if u != v:
                nbr[u].add(v); nbr[v].add(u)
    deg = {a: len(s) for a, s in nbr.items() if len(s) >= MIN_DEG}
    core = set(sorted(deg, key=deg.get, reverse=True)[:CORE_CAP])
    nbr2 = {a: (nbr[a] & core) for a in core}
    return {a: s for a, s in nbr2.items() if s}


def _events(nbr, seed, stealth_k=None):
    """stealth_k=None -> naiwny blast (fanout w jednym buckecie). stealth_k=K -> stealthy:
    <=K wyslan/nosiciela, PO JEDNYM na bucket (brak burstu nadawczego)."""
    rng = np.random.default_rng(seed)
    nodes = sorted(nbr)
    n_casc = max(20, len(nodes) // 30)
    casc = []
    seeds0 = [nodes[i] for i in rng.permutation(len(nodes))[:n_casc]]
    for s0 in seeds0:
        t0 = int(rng.integers(0, max(1, T_BUCKETS - MAX_HOPS - (stealth_k or 0))))
        if stealth_k is None:                                   # BLAST
            infected = {s0}; frontier = [s0]
            for hop in range(MAX_HOPS):
                nxt = []; bucket = (t0 + hop) % T_BUCKETS
                for u in frontier:
                    tg = sorted(nbr.get(u, ())); rng.shuffle(tg)
                    for v in tg[:MAX_FANOUT]:
                        if v in infected:
                            continue
                        casc.append((u, v, bucket, 1))
                        if rng.random() < P_INFECT:
                            infected.add(v); nxt.append(v)
                frontier = nxt
                if not frontier:
                    break
        else:                                                   # STEALTHY (1 wyslanie / bucket)
            infected = {s0}; frontier = [(s0, t0)]
            for _hop in range(MAX_HOPS):
                nxt = []
                for (u, tu) in frontier:
                    tg = sorted(nbr.get(u, ())); rng.shuffle(tg)
                    sent = 0
                    for v in tg:
                        if v in infected or sent >= stealth_k:
                            continue
                        bucket = (tu + 1 + sent) % T_BUCKETS
                        casc.append((u, v, bucket, 1)); sent += 1
                        if rng.random() < P_INFECT:
                            infected.add(v); nxt.append((v, bucket))
                frontier = nxt
                if not frontier:
                    break
    edges = [(u, w) for u in nbr for w in nbr[u]]
    benign = []
    for _ in range(3 * len(casc)):
        u, w = edges[int(rng.integers(len(edges)))]
        benign.append((u, w, int(rng.integers(T_BUCKETS)), 0))
    ev = casc + benign
    rng.shuffle(ev)
    return ev


def run(nbr, seed, stealth_k):
    torch.manual_seed(seed)
    events = _events(nbr, seed, stealth_k)
    recv_b, sent_b = _index_traffic(events)
    X1, Xc, _y = _features(events, nbr, recv_b, sent_b, seed)
    Xco = _compa_features(events)
    nodes = sorted({e[0] for e in events} | {e[1] for e in events})
    idx = {t: i for i, t in enumerate(nodes)}; n = len(nodes)
    y = np.array([e[3] for e in events], dtype=np.float32)
    vics = sorted({e[1] for e in events})
    rng = np.random.default_rng(seed)
    te_v = set(np.array(vics)[rng.permutation(len(vics))[:max(1, len(vics) // 3)]])
    te = np.array([e[1] in te_v for e in events]); tr = ~te
    out = {}
    for tag, X in (("tab_ctx", Xc), ("compa", Xco)):
        c = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(X[tr], y[tr])
        s = c.predict_proba(X[te])[:, 1]
        out[tag] = (roc_auc_score(y[te], s), _recall_at_fpr(y[te], s))
    Xn = _node_features(events, nbr, idx, n, recv_b, sent_b, tr=tr)
    s_idx = torch.tensor([idx[e[0]] for e in events])
    v_idx = torch.tensor([idx[e[1]] for e in events])
    ef = torch.tensor(X1, dtype=torch.float32)
    yt = torch.tensor(y); tr_t = torch.tensor(tr); te_t = torch.tensor(te)

    def train_temporal(pb):
        m = TemporalGNN(Xn.shape[1], X1.shape[1])
        opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
        lf = nn.BCEWithLogitsLoss()
        for _ in range(EPOCHS):
            m.train(); opt.zero_grad()
            loss = lf(m(Xn, pb, n, ef, s_idx, v_idx)[tr_t], yt[tr_t]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            s = torch.sigmoid(m(Xn, pb, n, ef, s_idx, v_idx))[te_t].numpy()
        return roc_auc_score(y[te], s), _recall_at_fpr(y[te], s)

    out["gnn_temporal"] = train_temporal(_per_bucket(events, idx))
    shuf = rng.integers(0, T_BUCKETS, size=len(events))
    ev_s = [(events[i][0], events[i][1], int(shuf[i]), events[i][3]) for i in range(len(events))]
    out["gnn_temporal_shuf"] = train_temporal(_per_bucket(ev_s, idx))
    return out, n, len(events), int(y.sum())


def main():
    rows = []
    for name, fname in DATASETS:
        nbr = load_graph(fname)
        for regime, sk in (("blast", None), ("stealthy", 2)):
            print(f"=== {name} [{regime}]: {len(nbr)} wezlow rdzenia ===", flush=True)
            acc = {}; n = ne = na = 0
            for seed in SEEDS:
                res, n, ne, na = run(nbr, seed, sk)
                for k, (a, r) in res.items():
                    acc.setdefault(k, {"a": [], "r": []})
                    acc[k]["a"].append(a); acc[k]["r"].append(r)
                print(f"  seed={seed} (wezly {n}, zdarz {ne}, ataki {na}): " + " | ".join(
                    f"{k} auc={v[0]:.3f} r1={v[1]:.3f}" for k, v in res.items()), flush=True)
            for k, v in acc.items():
                rows.append([name, regime, k, round(float(np.mean(v["a"])), 4),
                             round(float(np.mean(v["r"])), 4)])
    out = RESULTS / "exp_cascade_realgraphs.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["dataset", "regime", "model", "auc", "recall_fpr1"])
        w.writerows(rows)
    print(f"\n[realgraphs] -> {out}")


if __name__ == "__main__":
    main()
