"""P3 — adaptacyjny atak vs PANEL detektorow: ktore pekaja? (#GP-EXP-35).

Skladnik 'lamanie detektorow' (wymog top-venue). Ten sam adaptacyjny atakujacy (sterujacy zasiegiem K
i rozproszeniem g) wobec reprezentatywnych RODZIN detektorow:
  * 1-hop          : LightGBM na cechach lokalnych (slaby baseline)
  * COMPA          : wolumen/odchylenie per-konto [Egele i in.]
  * GCN-statyczny  : uczona reprezentacja grafu BEZ czasu
  * reczny-kontekst: hand-crafted kontekst kaskady (sygnal odbiorczy, reczny)
  * temporalny-GNN : nasz (sygnal odbiorczy, uczony)
Trzy strategie atakujacego: NAIWNA (duzy zasieg, brak ewazji), STEALTH-SPREAD (duzy zasieg + rozproszenie),
NISKI-ZASIEG. Teza: ewazja lamie wolumen/statyczny; sygnal odbiorczy (reczny+temporalny) trzyma.

Wyjscie: results/exp_p3_panel.csv
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

from exp_cascade import (
    T_BUCKETS,
    _contact_graph,
    _features,
    _index_traffic,  # noqa: E402
    _node_features,
    _recall_at_fpr,
    TemporalGNN,
    CascadeGNN,
    _norm_adj,
)
from exp_cascade_sweep import _compa_features  # noqa: E402
from exp_cascade_sweep import _event_batches as _per_bucket  # noqa: E402

SEEDS = list(range(3))
EPOCHS = 100
N_CASCADES = 30
MAX_HOPS = 4
P_INFECT = 0.6
# (nazwa, K fan-out, g rozproszenie)
STRATEGIES = [("naiwna", 8, 1), ("stealth-spread", 8, 8), ("niski-zasieg", 2, 1)]


def _simulate(seed, K, g):
    twins, nbr = _contact_graph()
    rng = np.random.default_rng(seed)
    events = []
    infected_all = set()
    seeds0 = [twins[i] for i in rng.permutation(len(twins))[:N_CASCADES]]
    for s0 in seeds0:
        t0 = int(rng.integers(0, T_BUCKETS))
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
                    events.append((u, v, (tu + 1 + sent * g) % T_BUCKETS, 1))
                    sent += 1
                    if rng.random() < P_INFECT:
                        infected.add(v)
                        nxt.append((v, (tu + 1 + sent * g) % T_BUCKETS))
            frontier = nxt
            if not frontier:
                break
        infected_all |= infected
    n_att = len(events)
    edges = [(u, w) for u in nbr for w in nbr[u]]
    for _ in range(3 * n_att):
        u, w = edges[int(rng.integers(len(edges)))]
        events.append((u, w, int(rng.integers(T_BUCKETS)), 0))
    rng.shuffle(events)
    return twins, nbr, events, len(infected_all)


def _panel(twins, nbr, events, seed):
    recv_b, sent_b = _index_traffic(events)
    X1, Xc, _y = _features(events, nbr, recv_b, sent_b, seed)
    Xco = _compa_features(events)
    idx = {t: i for i, t in enumerate(twins)}
    n = len(twins)
    y = np.array([e[3] for e in events], dtype=np.float32)
    vics = sorted({e[1] for e in events})
    rng = np.random.default_rng(seed)
    te_v = set(np.array(vics)[rng.permutation(len(vics))[: max(1, len(vics) // 3)]])
    te = np.array([e[1] in te_v for e in events])
    tr = ~te
    yte = y[te]
    out = {}
    for tag, X in (("1-hop", X1), ("COMPA", Xco), ("reczny-kontekst", Xc)):
        c = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(
            X[tr], y[tr]
        )
        out[tag] = roc_auc_score(yte, c.predict_proba(X[te])[:, 1])
    # GNN-y
    Xn = _node_features(events, nbr, idx, n, recv_b, sent_b, tr=tr)
    s_idx = torch.tensor([idx[e[0]] for e in events])
    v_idx = torch.tensor([idx[e[1]] for e in events])
    ef = torch.tensor(X1, dtype=torch.float32)
    yt = torch.tensor(y)
    tr_t = torch.tensor(tr)
    te_t = torch.tensor(te)
    lf = nn.BCEWithLogitsLoss()
    # statyczny GNN
    A = _norm_adj(nbr, idx, n)
    gs = CascadeGNN(Xn.shape[1], X1.shape[1])
    o = torch.optim.Adam(gs.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(EPOCHS):
        gs.train()
        o.zero_grad()
        lf(gs(A, Xn, s_idx, v_idx, ef)[tr_t], yt[tr_t]).backward()
        o.step()
    gs.eval()
    with torch.no_grad():
        out["GCN-statyczny"] = roc_auc_score(
            yte, torch.sigmoid(gs(A, Xn, s_idx, v_idx, ef))[te_t].numpy()
        )
    # temporalny GNN
    pb = _per_bucket(events, idx)
    gt = TemporalGNN(Xn.shape[1], X1.shape[1])
    o = torch.optim.Adam(gt.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(EPOCHS):
        gt.train()
        o.zero_grad()
        lf(gt(Xn, pb, n, ef, s_idx, v_idx)[tr_t], yt[tr_t]).backward()
        o.step()
    gt.eval()
    with torch.no_grad():
        out["temporalny-GNN"] = roc_auc_score(
            yte, torch.sigmoid(gt(Xn, pb, n, ef, s_idx, v_idx))[te_t].numpy()
        )
    return out


def main():
    n_nodes = len(_contact_graph()[0])
    order = ["1-hop", "COMPA", "GCN-statyczny", "reczny-kontekst", "temporalny-GNN"]
    agg = {s[0]: {k: [] for k in order} for s in STRATEGIES}
    reach = {}
    for name, K, g in STRATEGIES:
        infe = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            tw, nbr, ev, n_inf = _simulate(seed, K, g)
            infe.append(n_inf / n_nodes)
            for k, a in _panel(tw, nbr, ev, seed).items():
                agg[name][k].append(a)
        reach[name] = float(np.mean(infe))
        line = " | ".join(f"{k}={np.mean(agg[name][k]):.2f}" for k in order)
        print(f"  [{name}] zasieg {reach[name]:.0%}: {line}", flush=True)
    out = RESULTS / "exp_p3_panel.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strategia", "zasieg"] + order)
        for name, _K, _g in STRATEGIES:
            w.writerow(
                [name, round(reach[name], 3)]
                + [round(float(np.mean(agg[name][k])), 4) for k in order]
            )
    print(f"\n[p3-panel] -> {out}")


if __name__ == "__main__":
    main()
