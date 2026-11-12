"""Named baseline (COMPA-style) + sweepy wrazliwosci P_INFECT/glebokosc (#GP-EXP-23).

(a) BASELINE Z LITERATURY: detektor w stylu COMPA [Egele i in., NDSS'13] -- odchylenie od
    profilu per-konto (nowy odbiorca dla nadawcy + nietypowy czas nadawcy). Per-account, NIE
    widzi wielohopowej kaskady -> spodziewamy sie, ze przegra z temporalnym GNN.
(b) SWEEPY (jak EXP-12 w P1): P_INFECT in {0.3,0.5,0.7,0.9} i MAX_HOPS in {2,3,4,5}. Mierzymy
    tab_1hop / compa / tab_ctx / gnn_temporal -> kiedy przewaga temporalna jest duza, a kiedy znika.

Wyjscie: results/exp_cascade_sweep.csv
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
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
    _features,
    _index_traffic,  # noqa: E402
    _node_features,
    _recall_at_fpr,
    TemporalGNN,
)

SEEDS = list(range(8))
EPOCHS = 100
# Rozszerzony grid w DӣL: realne kampanie lateral maja per-odbiorca p_inf o rzedy nizsze niz 0.6
# (recenzja P2 #3) -> sprawdzamy, czy sygnal odbiorczy przezywa przy realistycznym, plytkim zasiegu.
P_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
H_GRID = [1, 2, 3, 4, 5]


def _compa_features(events):
    """COMPA-style: odchylenie od profilu per-konto. Cechy z OBSERWOWALNEGO strumienia:
    (i) nowosc odbiorcy dla nadawcy, (ii) nowosc czasu dla nadawcy, (iii) aktywnosc nadawcy.

    PRZYCZYNOWOSC NA POZIOMIE ZDARZENIA (fix po recenzji zewnetrznej): liczniki obejmuja
    wszystkie wczesniejsze kubełki ORAZ wczesniejsze zdarzenia z TEGO SAMEGO kubełka.
    Kolejnosc wewnatrz kubełka = kolejnosc na (tasowanej) liscie zdarzen, czyli losowa ale
    ustalona — bezstronna: kazde zdarzenie widzi w przyblizeniu polowe swoich kubełkowych
    sasiadow jako przeszlosc. Dzieki temu detektor wolumetryczny widzi rozwijajacy sie burst
    (podzial na poziomie kubełka calkowicie go przed nim ukrywal i czynil z niego slomiana
    kuke); kontekst odbiorczy i pamiec TGN pozostaja kubełkowo-przeszle [b-W, b-1]."""
    X = [None] * len(events)
    pair_c: dict = {}
    sb_c: dict = {}
    st_c: dict = {}
    order = sorted(
        range(len(events)), key=lambda i: events[i][2]
    )  # stabilnie po kubełku
    for i in order:
        s, v, b, _l = events[i]
        X[i] = [
            1.0
            / (1.0 + pair_c.get((s, v), 0)),  # nowosc odbiorcy (przeszlo zdarzeniowa)
            1.0 / (1.0 + sb_c.get((s, b), 0)),  # nowosc czasu
            float(st_c.get(s, 0)),  # aktywnosc nadawcy
        ]
        pair_c[(s, v)] = pair_c.get((s, v), 0) + 1
        sb_c[(s, b)] = sb_c.get((s, b), 0) + 1
        st_c[s] = st_c.get(s, 0) + 1
    return np.array(X, np.float32)


CHUNK = 64  # jak w cascadebench.models.event_batches


def _event_batches(events, idx, chunk=CHUNK):
    """Mini-batche zdarzen w porzadku (kubelek, pozycja na liscie) — przyczynowosc
    zdarzeniowa pamieci TGN (batch nie przekracza kubełka; batch jest wspolczesny)."""
    by_b = {}
    for i, e in enumerate(events):
        by_b.setdefault(e[2], []).append(i)
    out = []
    for b in sorted(by_b):
        g = by_b[b]
        for c in range(0, len(g), chunk):
            part = g[c : c + chunk]
            sb = torch.tensor([idx[events[i][0]] for i in part])
            vb = torch.tensor([idx[events[i][1]] for i in part])
            out.append((b, sb, vb, part))
    return out


def _per_bucket(events, idx):
    pb = []
    for b in range(T_BUCKETS):
        gidx = [i for i, e in enumerate(events) if e[2] == b]
        if not gidx:
            pb.append(
                (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), [])
            )
            continue
        sb = torch.tensor([idx[events[i][0]] for i in gidx])
        vb = torch.tensor([idx[events[i][1]] for i in gidx])
        pb.append((sb, vb, gidx))
    return pb


def _eval_point(seed):
    """Jeden przebieg na biezacych XC.P_INFECT/XC.MAX_HOPS. Zwraca dict model->(auc,r1)."""
    torch.manual_seed(seed)
    twins, nbr, events = XC._simulate(seed)
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

    out = {}
    for tag, X in (("tab_1hop", X1), ("compa", Xco), ("tab_ctx", Xc)):
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
    return out


def _sweep(var, values, set_fn, rows, perseed):
    p0, h0 = XC.P_INFECT, XC.MAX_HOPS
    for val in values:
        set_fn(val)
        acc = {}
        for seed in SEEDS:
            for k, (a, r) in _eval_point(seed).items():
                acc.setdefault(k, {"a": [], "r": []})
                acc[k]["a"].append(a)
                acc[k]["r"].append(r)
                # R4-A4: per-seed dump (bootstrap CI w tabeli S2)
                perseed.append([var, val, seed, k, round(a, 4), round(r, 4)])
        for k, v in acc.items():
            rows.append(
                [
                    var,
                    val,
                    k,
                    round(float(np.mean(v["a"])), 4),
                    round(float(np.mean(v["r"])), 4),
                ]
            )
        line = " | ".join(f"{k} {np.mean(v['a']):.3f}" for k, v in acc.items())
        print(f"  {var}={val}: {line}", flush=True)
    XC.P_INFECT, XC.MAX_HOPS = p0, h0


def main():
    rows = []
    perseed = []
    print("=== sweep P_INFECT (MAX_HOPS=4) ===", flush=True)
    XC.MAX_HOPS = 4
    _sweep("p_infect", P_GRID, lambda v: setattr(XC, "P_INFECT", v), rows, perseed)
    print("=== sweep MAX_HOPS (P_INFECT=0.6) ===", flush=True)
    XC.P_INFECT = 0.6
    _sweep("max_hops", H_GRID, lambda v: setattr(XC, "MAX_HOPS", v), rows, perseed)
    out = RESULTS / "exp_cascade_sweep.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sweep", "value", "model", "auc", "recall_fpr1"])
        w.writerows(rows)
    print(f"\n[sweep] -> {out}")
    out_ps = RESULTS / "exp_cascade_sweep_perseed.csv"
    with out_ps.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sweep", "value", "seed", "model", "auc", "recall_fpr1"])
        w.writerows(perseed)
    print(f"[sweep-perseed] -> {out_ps}")


if __name__ == "__main__":
    main()
