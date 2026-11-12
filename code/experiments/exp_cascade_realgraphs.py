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

from exp_cascade import (
    T_BUCKETS,
    _features,
    _index_traffic,  # noqa: E402
    _node_features,
    _recall_at_fpr,
    TemporalGNN,
)
from exp_cascade_sweep import _compa_features, _event_batches  # noqa: E402

SEEDS = list(range(10))  # R4-A2: >=10 ziaren na graf
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
                nbr[u].add(v)
                nbr[v].add(u)
    deg = {a: len(s) for a, s in nbr.items() if len(s) >= MIN_DEG}
    core = set(sorted(deg, key=deg.get, reverse=True)[:CORE_CAP])
    nbr2 = {a: (nbr[a] & core) for a in core}
    return {a: s for a, s in nbr2.items() if s}


def _benign_gap_pool(nbr, seed):
    """R4-A2: empiryczny rozklad odstepow tla realnego grafu — odstepy miedzy
    kolejnymi wysylaniami tego samego nadawcy w matched benign (kubelki jednostajne)."""
    rng = np.random.default_rng(seed + 555)
    edges = [(u, w) for u in nbr for w in nbr[u]]
    by_sender = {}
    for _ in range(3000):
        u, w = edges[int(rng.integers(len(edges)))]
        b = int(rng.integers(T_BUCKETS))
        by_sender.setdefault(u, []).append(b)
    pool = []
    for bs in by_sender.values():
        bs.sort()
        pool.extend(b2 - b1 for b1, b2 in zip(bs, bs[1:]) if b2 > b1)
    return pool or [1]


def _events(nbr, seed, stealth_k=None, timing="det"):
    """stealth_k=None -> naiwny blast (fanout w jednym buckecie). stealth_k=K -> stealthy:
    <=K wyslan/nosiciela, PO JEDNYM na bucket (brak burstu nadawczego)."""
    rng = np.random.default_rng(seed)
    nodes = sorted(nbr)
    n_casc = max(20, len(nodes) // 30)
    casc = []
    seeds0 = [nodes[i] for i in rng.permutation(len(nodes))[:n_casc]]
    for s0 in seeds0:
        t0 = int(rng.integers(0, max(1, T_BUCKETS - MAX_HOPS - (stealth_k or 0))))
        if stealth_k is None:  # BLAST
            infected = {s0}
            frontier = [s0]
            for hop in range(MAX_HOPS):
                nxt = []
                bucket = (t0 + hop) % T_BUCKETS
                for u in frontier:
                    tg = sorted(nbr.get(u, ()))
                    rng.shuffle(tg)
                    for v in tg[:MAX_FANOUT]:
                        if v in infected:
                            continue
                        casc.append((u, v, bucket, 1))
                        if rng.random() < P_INFECT:
                            infected.add(v)
                            nxt.append(v)
                frontier = nxt
                if not frontier:
                    break
        else:  # STEALTHY (1 wyslanie / bucket; emp: odstepy z rozkladu tla)
            gap_pool = _benign_gap_pool(nbr, seed) if timing == "emp" else None
            infected = {s0}
            frontier = [(s0, t0)]
            for _hop in range(MAX_HOPS):
                nxt = []
                for u, tu in frontier:
                    tg = sorted(nbr.get(u, ()))
                    rng.shuffle(tg)
                    if timing == "emp" and gap_pool:
                        gaps = [0] + [
                            gap_pool[int(rng.integers(len(gap_pool)))]
                            for _ in range(stealth_k)
                        ]
                        cum = np.cumsum(gaps)
                    sent = 0
                    for v in tg:
                        if v in infected or sent >= stealth_k:
                            continue
                        if timing == "emp" and gap_pool:
                            bucket = (tu + int(cum[sent])) % T_BUCKETS
                        else:
                            bucket = (tu + 1 + sent) % T_BUCKETS
                        casc.append((u, v, bucket, 1))
                        sent += 1
                        if rng.random() < P_INFECT:
                            infected.add(v)
                            nxt.append((v, bucket))
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


def _timing_stats(events):
    """R4-A2: KS odstepow atak vs tlo + sonda czysto czasowa (per seed, in-run)."""
    from scipy.stats import ks_2samp

    from exp_injection_probe import _gaps, _probe_auc

    a_g, _ = _gaps(events, 1)
    b_g, _ = _gaps(events, 0)
    if len(a_g) >= 5 and len(b_g) >= 5:
        ks, p = ks_2samp(a_g, b_g)
        ks = float(ks)
    else:
        ks, p = float("nan"), float("nan")
    probe = _probe_auc(events, T_BUCKETS, 123)
    return ks, float(p), probe


def run(nbr, seed, stealth_k, timing="det"):
    torch.manual_seed(seed)
    events = _events(nbr, seed, stealth_k, timing)
    recv_b, sent_b = _index_traffic(events)
    X1, Xc, _y = _features(events, nbr, recv_b, sent_b, seed)
    Xco = _compa_features(events)
    nodes = sorted({e[0] for e in events} | {e[1] for e in events})
    idx = {t: i for i, t in enumerate(nodes)}
    n = len(nodes)
    y = np.array([e[3] for e in events], dtype=np.float32)
    vics = sorted({e[1] for e in events})
    rng = np.random.default_rng(seed)
    te_v = set(np.array(vics)[rng.permutation(len(vics))[: max(1, len(vics) // 3)]])
    te = np.array([e[1] in te_v for e in events])
    tr = ~te
    out = {}
    for tag, X in (("tab_ctx", Xc), ("compa", Xco)):
        c = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(
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

    def train_temporal(pb):
        m = TemporalGNN(Xn.shape[1], X1.shape[1])
        opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
        lf = nn.BCEWithLogitsLoss()
        for _ in range(EPOCHS):
            m.train()
            opt.zero_grad()
            loss = lf(m(Xn, pb, n, ef, s_idx, v_idx)[tr_t], yt[tr_t])
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            s = torch.sigmoid(m(Xn, pb, n, ef, s_idx, v_idx))[te_t].numpy()
        return roc_auc_score(y[te], s), _recall_at_fpr(y[te], s)

    out["gnn_temporal"] = train_temporal(_event_batches(events, idx))
    shuf = rng.integers(0, T_BUCKETS, size=len(events))
    ev_s = [
        (events[i][0], events[i][1], int(shuf[i]), events[i][3])
        for i in range(len(events))
    ]
    out["gnn_temporal_shuf"] = train_temporal(_event_batches(ev_s, idx))
    ks, kp, probe = _timing_stats(events)
    return out, n, len(events), int(y.sum()), ks, probe


def main():
    # R4: opcjonalny filtr datasetu (rownoleglosc: python exp_cascade_realgraphs.py --only email-Eu-core)
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    rows = []
    perseed = []
    trows = []
    for name, fname in DATASETS:
        if only and name != only:
            continue
        nbr = load_graph(fname)
        # R4-A2: blast tylko det; stealthy w OBU wariantach czasowania (det/emp)
        for regime, sk, timing in (
            ("blast", None, "det"),
            ("stealthy", 2, "det"),
            ("stealthy", 2, "emp"),
        ):
            print(
                f"=== {name} [{regime}/{timing}]: {len(nbr)} wezlow rdzenia ===",
                flush=True,
            )
            acc = {}
            n = ne = na = 0
            ks_l, pr_l = [], []
            for seed in SEEDS:
                res, n, ne, na, ks, probe = run(nbr, seed, sk, timing)
                ks_l.append(ks)
                pr_l.append(probe)
                for k, (a, r) in res.items():
                    acc.setdefault(k, {"a": [], "r": []})
                    acc[k]["a"].append(a)
                    acc[k]["r"].append(r)
                    perseed.append(
                        [name, regime, timing, seed, k, round(a, 4), round(r, 4)]
                    )
                print(
                    f"  seed={seed} (wezly {n}, zdarz {ne}, ataki {na}): "
                    + " | ".join(
                        f"{k} auc={v[0]:.3f} r1={v[1]:.3f}" for k, v in res.items()
                    )
                    + f" | KS={ks:.2f} probe={probe:.2f}",
                    flush=True,
                )
            for k, v in acc.items():
                rows.append(
                    [
                        name,
                        regime,
                        timing,
                        k,
                        round(float(np.mean(v["a"])), 4),
                        round(float(np.mean(v["r"])), 4),
                    ]
                )
            trows.append(
                [
                    name,
                    regime,
                    timing,
                    round(float(np.nanmean(ks_l)), 4),
                    round(float(np.mean(pr_l)), 4),
                ]
            )
    suffix = ("_" + only) if only else ""
    out = RESULTS / f"exp_cascade_realgraphs{suffix}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "regime", "timing", "model", "auc", "recall_fpr1"])
        w.writerows(rows)
    out_t = RESULTS / f"exp_cascade_realgraphs_timing{suffix}.csv"
    with out_t.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "regime", "timing", "ks_mean", "probe_auc"])
        w.writerows(trows)
    print(f"\n[realgraphs] -> {out}\n[realgraphs-timing] -> {out_t}")
    out_ps = RESULTS / f"exp_cascade_realgraphs{suffix}_perseed.csv"
    with out_ps.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["dataset", "regime", "timing", "seed", "model", "auc", "recall_fpr1"]
        )
        w.writerows(perseed)
    print(f"[realgraphs-perseed] -> {out_ps}")


if __name__ == "__main__":
    main()
