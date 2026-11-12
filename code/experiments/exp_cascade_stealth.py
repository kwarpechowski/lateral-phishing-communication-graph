"""Stealthy attacker: czy temporalny GNN wykrywa kaskade, gdy atak UNIKA detekcji wolumenowej (#GP-EXP-24).

Named baseline COMPA bil nasz GNN, bo generator blastowal wszystkie kontakty w jednym buckecie
(ekstremalny burst nadawczy). Realny lateral phisher tego unika (Ho i in.). Tu STEALTHY atak:
zainfekowany wezel wysyla do <=K kontaktow, PO JEDNYM na bucket (brak burstu, maly wolumen per-konto)
-> COMPA traci sygnal nadawczy. Pytanie: czy kontekst ODBIORCZY kaskady (temporalny GNN) wciaz dziala?

Sweep po K (poziom ukrycia): K in {1,2,3,5,unlimited}. Crossover COMPA (spada) vs gnn_temporal.
Wyjscie: results/exp_cascade_stealth.csv
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

import exp_cascade as XC  # noqa: E402
from exp_cascade import (
    T_BUCKETS,
    _contact_graph,
    _features,
    _index_traffic,  # noqa: E402
    _node_features,
    _recall_at_fpr,
    TemporalGNN,
)
from exp_cascade_sweep import _compa_features, _event_batches  # noqa: E402

SEEDS = list(range(20))  # 20 ziaren (recenzja r1 pkt 13: stealth/baselines = 20 seedow)
EPOCHS = 100
K_GRID = [1, 2, 3, 5, 999]  # 999 = bez ograniczenia (blast, jak dotad)
N_CASCADES = 30
MAX_HOPS = 4
P_INFECT = 0.6


def _simulate_stealth(seed, K):
    """Kaskada stealthy: <=K wyslan na nosiciela, po jednym na bucket (brak burstu)."""
    twins, nbr = _contact_graph()
    rng = np.random.default_rng(seed)
    events = []
    seeds0 = [twins[i] for i in rng.permutation(len(twins))[:N_CASCADES]]
    for s0 in seeds0:
        t0 = int(rng.integers(0, max(1, T_BUCKETS - MAX_HOPS - K)))
        infected = {s0}
        frontier = [(s0, t0)]
        for _hop in range(MAX_HOPS):
            nxt = []
            for u, tu in frontier:
                tg = sorted(nbr.get(u, ()))
                rng.shuffle(tg)
                sent = 0
                for v in tg:
                    if v in infected or sent >= K:
                        continue
                    bucket = (
                        tu + 1 + sent
                    ) % T_BUCKETS  # rozproszenie: 1 wyslanie / bucket
                    events.append((u, v, bucket, 1))
                    sent += 1
                    if rng.random() < P_INFECT:
                        infected.add(v)
                        nxt.append((v, bucket))
            frontier = nxt
            if not frontier:
                break
    n_att = sum(1 for e in events if e[3] == 1)
    edges = [(u, w) for u in nbr for w in nbr[u]]
    for _ in range(3 * n_att):  # benign: dopasowane krawedzie, rozproszone
        u, w = edges[int(rng.integers(len(edges)))]
        events.append((u, w, int(rng.integers(T_BUCKETS)), 0))
    rng.shuffle(events)
    return twins, nbr, events


def _tc_stats(events, seed):
    """Udzial zdarzen OFF-rhythm (tc=0) per klasa — audyt bitu rytmu (recenzja r2, pkt 3)."""
    from exp_cascade import is_consistent

    att = [e for e in events if e[3] == 1]
    ben = [e for e in events if e[3] == 0]
    off = lambda ev: (
        sum(1 for s, _v, b, _l in ev if not is_consistent(s, b, seed)) / max(1, len(ev))
    )
    return off(att), off(ben)


def _eval(seed, K):
    torch.manual_seed(seed)
    twins, nbr, events = _simulate_stealth(seed, K)
    recv_b, sent_b = _index_traffic(events)
    X1, Xc, _y = _features(events, nbr, recv_b, sent_b, seed)
    Xco = _compa_features(events)
    from exp_cascade import is_consistent

    Xro = np.array(
        [[0.0 if is_consistent(s, b, seed) else 1.0] for s, _v, b, _l in events],
        dtype=np.float32,
    )
    idx = {t: i for i, t in enumerate(twins)}
    n = len(twins)
    y = np.array([e[3] for e in events], dtype=np.float32)
    vics = sorted({e[1] for e in events})
    rng = np.random.default_rng(seed)
    te_v = set(np.array(vics)[rng.permutation(len(vics))[: max(1, len(vics) // 3)]])
    te = np.array([e[1] in te_v for e in events])
    tr = ~te
    out = {}
    for tag, X in (
        ("compa", Xco),
        ("tab_ctx", Xc),
        ("tab_1hop", X1),
        ("rhythm_only", Xro),
    ):
        c = LGBMClassifier(random_state=42, n_estimators=200, verbose=-1).fit(
            X[tr], y[tr]
        )
        s = c.predict_proba(X[te])[:, 1]
        out[tag] = (roc_auc_score(y[te], s), _recall_at_fpr(y[te], s))
    Xn = _node_features(events, nbr, idx, n, recv_b, sent_b, tr=tr)
    s_idx = torch.tensor([idx[e[0]] for e in events])
    v_idx = torch.tensor([idx[e[1]] for e in events])
    ef = torch.tensor(X1, dtype=torch.float32)
    yt = torch.tensor(y)
    tr_t = torch.tensor(tr)
    te_t = torch.tensor(te)
    pb = _event_batches(events, idx)
    m = TemporalGNN(Xn.shape[1], X1.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        m.train()
        opt.zero_grad()
        loss = lossf(m(Xn, pb, n, ef, s_idx, v_idx)[tr_t], yt[tr_t])
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        sg = torch.sigmoid(m(Xn, pb, n, ef, s_idx, v_idx))[te_t].numpy()
    out["gnn_temporal"] = (roc_auc_score(y[te], sg), _recall_at_fpr(y[te], sg))
    off_att, off_ben = _tc_stats(events, seed)
    return out, int(y.sum()), len(y), off_att, off_ben


def main():
    torch.set_num_threads(4)
    rows = []
    tc_rows = []
    perseed = []
    for K in K_GRID:
        acc = {}
        na = nt = 0
        off_a, off_b = [], []
        for seed in SEEDS:
            res, na, nt, oa, ob = _eval(seed, K)
            off_a.append(oa)
            off_b.append(ob)
            for k, (a, r) in res.items():
                perseed.append([K, seed, k, round(a, 4), round(r, 4)])
            for k, (a, r) in res.items():
                acc.setdefault(k, {"a": [], "r": []})
                acc[k]["a"].append(a)
                acc[k]["r"].append(r)
        for k, v in acc.items():
            rows.append(
                [
                    K,
                    k,
                    round(float(np.mean(v["a"])), 4),
                    round(float(np.mean(v["r"])), 4),
                ]
            )
        tc_rows.append(
            [K, round(float(np.mean(off_a)), 4), round(float(np.mean(off_b)), 4)]
        )
        lab = "blast" if K == 999 else f"K={K}"
        print(
            f"  {lab} (ataki~{na}/{nt}): "
            + " | ".join(
                f"{k} auc={np.mean(v['a']):.3f} r1={np.mean(v['r']):.3f}"
                for k, v in acc.items()
            ),
            flush=True,
        )
        print(
            f"     off-rhythm: ataki {np.mean(off_a):.2f} vs benign {np.mean(off_b):.2f}",
            flush=True,
        )
    out = RESULTS / "exp_cascade_stealth.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["K", "model", "auc", "recall_fpr1"])
        w.writerows(rows)
    out_tc = RESULTS / "exp_cascade_stealth_tc.csv"
    with out_tc.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["K", "off_rhythm_attacks", "off_rhythm_benign"])
        w.writerows(tc_rows)
    print(f"\n[stealth] -> {out}\n[stealth-tc] -> {out_tc}")
    out_ps = RESULTS / "exp_cascade_stealth_perseed.csv"
    with out_ps.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["K", "seed", "model", "auc", "recall_fpr1"])
        w.writerows(perseed)
    print(f"[stealth-perseed] -> {out_ps}")


if __name__ == "__main__":
    main()
