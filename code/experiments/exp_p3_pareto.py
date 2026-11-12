"""P3 (adwersaryjna odporność) — front Pareto ewazja vs zasieg (#GP-EXP-33).

Rdzen P3: czy atakujacy moze JEDNOCZESNIE uniknac wykrycia I zachowac zasieg? Atakujacy steruje
ukryciem (fan-out K = ile kontaktow zaraza nosiciel, rozproszone w czasie). Mierzymy NARAZ:
  * wykrywalnosc : nasz temporalny GNN (AUC, Recall@FPR1%) -- im stealthy, tym nizsza
  * zasieg        : # zdarzen ataku i # zarazonych wezlow -- im stealthy, tym mniejszy
Hipoteza (odpornosc): wykrywalnosc i zasieg sa DODATNIO skorelowane -> atakujacy nie ma obu naraz;
ewazja kosztuje zasieg. To charakteryzuje odpornosc detektora w operacyjnie istotnym rezimie.

Wyjscie: results/exp_p3_pareto.csv
"""
from __future__ import annotations

import csv
import sys
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

from exp_cascade import (T_BUCKETS, _contact_graph, _features, _index_traffic,   # noqa: E402
                         _node_features, _recall_at_fpr, TemporalGNN)
from exp_cascade_sweep import _event_batches as _per_bucket                             # noqa: E402

SEEDS = list(range(3))
EPOCHS = 100
K_GRID = [1, 2, 3, 5, 8, 999]            # ukrycie: fan-out na nosiciela (999 = bez limitu)
N_CASCADES = 30
MAX_HOPS = 4
P_INFECT = 0.6


def _simulate(seed, K):
    """Kaskada o fan-oucie K (1 wyslanie/bucket). Zwraca events + miary zasiegu."""
    twins, nbr = _contact_graph()
    rng = np.random.default_rng(seed)
    events = []; infected_all = set()
    seeds0 = [twins[i] for i in rng.permutation(len(twins))[:N_CASCADES]]
    for s0 in seeds0:
        t0 = int(rng.integers(0, max(1, T_BUCKETS - MAX_HOPS - (K if K < 999 else 0))))
        infected = {s0}; frontier = [(s0, t0)]
        for _hop in range(MAX_HOPS):
            nxt = []
            for (u, tu) in frontier:
                tg = sorted(nbr.get(u, ())); rng.shuffle(tg); sent = 0
                for v in tg:
                    if v in infected or (K < 999 and sent >= K):
                        continue
                    bucket = (tu + 1 + sent) % T_BUCKETS
                    events.append((u, v, bucket, 1)); sent += 1
                    if rng.random() < P_INFECT:
                        infected.add(v); nxt.append((v, bucket))
            frontier = nxt
            if not frontier:
                break
        infected_all |= infected
    n_att = len(events)
    # benign dopasowany (jak w stealth/realgraphs)
    edges = [(u, w) for u in nbr for w in nbr[u]]
    for _ in range(3 * n_att):
        u, w = edges[int(rng.integers(len(edges)))]
        events.append((u, w, int(rng.integers(T_BUCKETS)), 0))
    rng.shuffle(events)
    return twins, nbr, events, n_att, len(infected_all)


def _detect(twins, nbr, events, seed):
    recv_b, sent_b = _index_traffic(events)
    X1, Xc, _y = _features(events, nbr, recv_b, sent_b, seed)
    idx = {t: i for i, t in enumerate(twins)}; n = len(twins)
    y = np.array([e[3] for e in events], dtype=np.float32)
    vics = sorted({e[1] for e in events})
    rng = np.random.default_rng(seed)
    te_v = set(np.array(vics)[rng.permutation(len(vics))[:max(1, len(vics) // 3)]])
    te = np.array([e[1] in te_v for e in events]); tr = ~te
    Xn = _node_features(events, nbr, idx, n, recv_b, sent_b, tr=tr)
    s_idx = torch.tensor([idx[e[0]] for e in events]); v_idx = torch.tensor([idx[e[1]] for e in events])
    ef = torch.tensor(X1, dtype=torch.float32); yt = torch.tensor(y); tr_t = torch.tensor(tr)
    m = TemporalGNN(Xn.shape[1], X1.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4); lf = nn.BCEWithLogitsLoss()
    pb = _per_bucket(events, idx)
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        loss = lf(m(Xn, pb, n, ef, s_idx, v_idx)[tr_t], yt[tr_t]); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        sc = torch.sigmoid(m(Xn, pb, n, ef, s_idx, v_idx))[torch.tensor(te)].numpy()
    yte = y[te]
    return roc_auc_score(yte, sc), _recall_at_fpr(yte, sc)


def main():
    rows = []
    n_nodes = len(_contact_graph()[0])
    for K in K_GRID:
        aucs, recs, reach, infe = [], [], [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            tw, nbr, ev, n_att, n_inf = _simulate(seed, K)
            a, r = _detect(tw, nbr, ev, seed)
            aucs.append(a); recs.append(r); reach.append(n_att); infe.append(n_inf)
        lab = "blast" if K == 999 else f"K={K}"
        row = [K, round(float(np.mean(aucs)), 4), round(float(np.mean(recs)), 4),
               int(np.mean(reach)), round(float(np.mean(infe)) / n_nodes, 4)]
        rows.append(row)
        print(f"  {lab}: detekcja AUC={row[1]:.3f} R@1%={row[2]:.3f} | "
              f"zasieg: {row[3]} zdarzen, {row[4]:.0%} grafu zarazone", flush=True)
    out = RESULTS / "exp_p3_pareto.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["K", "auc", "recall_fpr1", "reach_events", "infected_frac"])
        w.writerows(rows)
    print(f"\n[p3-pareto] -> {out}")
    print("[p3-pareto] WNIOSEK: jesli detekcja ROSNIE z zasiegiem -> atakujacy nie ma obu naraz.")


if __name__ == "__main__":
    main()
