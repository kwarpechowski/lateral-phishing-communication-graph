# -*- coding: utf-8 -*-
"""PRAWDZIWE KASKADY JAKO TRUDNE NEGATYWY (#GP-EXP-27, fix po recenzji zewnetrznej).

Recenzent: dane zawieraja tez legalne fale (newslettery, ogloszenia HR, incident
comms). Detektor kaskadowy nie moze traktowac kazdej fali jako ataku — rozdzielenie
"fala" od "fala zlosliwa" musi przyjsc z cechami odbiorczymi (rytm, typowosc odbiorcy),
nie z samego faktu propagacji.

Protokol (framework-1, zgodny z exp_cascade):
  * graf kontaktow: ten sam co exp_cascade (build_layers, 600 twins)
  * ATAK (label=1): identyczny protokol jak _simulate w exp_cascade (P_INFECT, MAX_HOPS)
  * FALE LEGALNE (label=0, TRUDNE NEGATYWY): ~30% liczby kaskad atakowych; lagodniejsze
    — fanout <=2, p_infect=0.35, 3 hopy, hop co 2 buckety (wolniejszy rytm); nadawcy
    "zaufani" (spokojny rytm is_consistent). Strukturalnie SA fala — punktowko
    niewiniaci (f_s_in_contacts) bo rozchodza sie po kontaktach.
  * benign punktowy: bez zmian (BENIGN_PER_BUCKET)
  * warunki: baseline (bez fal legalnych) vs legal (z falami) — spadek AUC mierzy
    podatnosc detektora na trudne negatywy.

Panel: COMPA (event-causal), reczny-kontekst, temporalny-GNN. 20 ziaren, per-seed dump.
Wyjscie: results/exp_legal_cascade.csv + _perseed.csv
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

sys.argv = [sys.argv[0]]  # wyczysc argv przed importem exp_cascade (SPLIT/SCALE_N)

from exp_cascade import (
    T_BUCKETS,
    N_CASCADES,
    MAX_HOPS,
    P_INFECT,  # noqa: E402
    BENIGN_PER_BUCKET,
    _contact_graph,
    _features,
    _index_traffic,
    _node_features,
    _recall_at_fpr,
    TemporalGNN,
    is_consistent,
)
from exp_cascade_sweep import _compa_features, _per_bucket  # noqa: E402

SEEDS = list(range(20))
EPOCHS = 100
LEGAL_FRAC = 0.30  # ~30% liczby kasc. atakowych -> legalne fale
LEGAL_FANOUT = 2  # lagodniejszy fanout niz atak (10)
LEGAL_P_INFECT = 0.35
LEGAL_HOPS = 3
LEGAL_STEP = 2  # hop co 2 buckety (wolniejszy rytm niz atak: co 1)


def _simulate(seed, with_legal):
    """Atak jak w exp_cascade + (opcjonalnie) legalne fale jako trudne negatywy."""
    twins, nbr = _contact_graph()
    rng = np.random.default_rng(seed)
    events = []

    # --- atak (label=1): identyczny protokol jak exp_cascade._simulate ---
    seeds0 = [twins[i] for i in rng.permutation(len(twins))[:N_CASCADES]]
    for s0 in seeds0:
        t0 = int(rng.integers(0, T_BUCKETS - MAX_HOPS))
        infected = {s0}
        frontier = [s0]
        for hop in range(MAX_HOPS):
            nxt = []
            bucket = t0 + hop
            for u in frontier:
                targets = sorted(nbr.get(u, ()))
                rng.shuffle(targets)
                for v in targets:
                    if v in infected:
                        continue
                    events.append((u, v, bucket, 1))
                    if rng.random() < P_INFECT:
                        infected.add(v)
                        nxt.append(v)
            frontier = nxt
            if not frontier:
                break

    # --- legalne fale (label=0): strukturalnie fala, ale lagodna i w rytmie ---
    if with_legal:
        n_legal = max(2, int(round(LEGAL_FRAC * N_CASCADES)))
        lseeds = [twins[i] for i in rng.permutation(len(twins))[:n_legal]]
        for s0 in lseeds:
            t0 = int(rng.integers(0, T_BUCKETS - LEGAL_HOPS * LEGAL_STEP))
            seen = {s0}
            frontier = [s0]
            for hop in range(LEGAL_HOPS):
                nxt = []
                bucket = t0 + hop * LEGAL_STEP
                for u in frontier:
                    # nadawca fali legalnej jest w rytmie (typowy nadawca)
                    if not is_consistent(u, bucket, seed):
                        continue
                    targets = sorted(nbr.get(u, ()))
                    rng.shuffle(targets)
                    sent = 0
                    for v in targets:
                        if v in seen or sent >= LEGAL_FANOUT:
                            continue
                        events.append((u, v, bucket, 0))
                        sent += 1
                        if rng.random() < LEGAL_P_INFECT:
                            seen.add(v)
                            nxt.append(v)
                frontier = nxt
                if not frontier:
                    break

    # --- benign punktowy (label=0): znany kontakt, w rytmie ---
    for bucket in range(T_BUCKETS):
        for _ in range(BENIGN_PER_BUCKET):
            v = twins[int(rng.integers(len(twins)))]
            cs = sorted(nbr.get(v, ()))
            if not cs:
                continue
            s = cs[int(rng.integers(len(cs)))]
            events.append((s, v, bucket, 0))
    rng.shuffle(events)
    return twins, nbr, events


def run_seed(seed, with_legal):
    torch.manual_seed(seed)
    twins, nbr, events = _simulate(seed, with_legal)
    recv_b, sent_b = _index_traffic(events)
    X1, Xc, y = _features(events, nbr, recv_b, sent_b, seed)
    Xco = _compa_features(events)
    idx = {t: i for i, t in enumerate(twins)}
    n = len(twins)
    ya = np.asarray(y, dtype=np.float32)

    # split po ofiarach (jak exp_cascade domyslnie)
    vics = sorted({e[1] for e in events})
    rng = np.random.default_rng(seed)
    te_v = set(np.array(vics)[rng.permutation(len(vics))[: max(1, len(vics) // 3)]])
    te = np.array([e[1] in te_v for e in events])
    tr = ~te

    out = {}
    for tag, X in (("compa", Xco), ("reczny-kontekst", Xc)):
        c = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(
            X[tr], ya[tr]
        )
        s = c.predict_proba(X[te])[:, 1]
        out[tag] = (roc_auc_score(ya[te], s), _recall_at_fpr(ya[te], s))

    Xn = _node_features(events, nbr, idx, n, recv_b, sent_b, tr=tr)
    s_idx = torch.tensor([idx[e[0]] for e in events])
    v_idx = torch.tensor([idx[e[1]] for e in events])
    ef = torch.tensor(X1, dtype=torch.float32)
    yt = torch.tensor(ya)
    tr_t = torch.tensor(tr)
    te_t = torch.tensor(te)

    m = TemporalGNN(Xn.shape[1], X1.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    lf = nn.BCEWithLogitsLoss()
    pb = _per_bucket(events, idx)
    for _ in range(EPOCHS):
        m.train()
        opt.zero_grad()
        loss = lf(m(Xn, pb, n, ef, s_idx, v_idx)[tr_t], yt[tr_t])
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        s = torch.sigmoid(m(Xn, pb, n, ef, s_idx, v_idx))[te_t].numpy()
    out["temporalny-GNN"] = (roc_auc_score(ya[te], s), _recall_at_fpr(ya[te], s))
    return out


def main():
    torch.set_num_threads(4)
    rows, perseed = [], []
    for cond, with_legal in (("baseline", False), ("legal-waves", True)):
        acc = {}
        print(f"=== [legal-cascade] warunek: {cond} ===", flush=True)
        for seed in SEEDS:
            res = run_seed(seed, with_legal)
            for k, (a, r) in res.items():
                acc.setdefault(k, {"a": [], "r": []})
                acc[k]["a"].append(a)
                acc[k]["r"].append(r)
                perseed.append([cond, seed, k, round(a, 4), round(r, 4)])
            print(
                f"  seed={seed}: "
                + " | ".join(f"{k}={v[0]:.3f}" for k, v in res.items()),
                flush=True,
            )
        for k, v in acc.items():
            rows.append(
                [
                    cond,
                    k,
                    round(float(np.mean(v["a"])), 4),
                    round(float(np.std(v["a"])), 4),
                    round(float(np.mean(v["r"])), 4),
                    len(v["a"]),
                ]
            )
    out = RESULTS / "exp_legal_cascade.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["warunek", "model", "auc_mean", "auc_sd", "recall_fpr1", "n"])
        w.writerows(rows)
    out_ps = RESULTS / "exp_legal_cascade_perseed.csv"
    with out_ps.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["warunek", "seed", "model", "auc", "recall_fpr1"])
        w.writerows(perseed)
    print(f"\n[legal-cascade] -> {out}\n[legal-cascade-perseed] -> {out_ps}")


if __name__ == "__main__":
    main()
