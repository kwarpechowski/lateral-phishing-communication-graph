# -*- coding: utf-8 -*-
"""Testy jednostkowe anty-wyciek (przyczynowosc zdarzeniowa, zadanie A1).

ZASADA: cechy/logity zdarzenia e musza byc IDENTYCZNE po usunieciu ze strumienia
wszystkich zdarzen o czasie >= t(e), gdzie t(e) = (kubelek, pozycja na liscie).

Uruchomienie:  python tests/test_causality.py   (bez pytest — zwykle asercje)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "experiments"))

RESULTS = []

# ---- wspolny mini-graf + strumien ----
NODES = list("abcdefghij")
NBR = {u: set(NODES[(i + 1) % 10]) | {NODES[(i + 3) % 10]} for i, u in enumerate(NODES)}
# zdarzenia (s, v, bucket, label); celowo nieposortowane (tasowanie)
EVENTS = [
    ("a", "b", 3, 0),
    ("b", "c", 1, 0),
    ("c", "a", 2, 0),
    ("d", "e", 2, 1),
    ("a", "d", 1, 1),
    ("e", "f", 3, 0),
    ("f", "g", 0, 0),
    ("g", "h", 3, 1),
    ("h", "a", 2, 1),
    ("b", "f", 3, 1),
    ("c", "e", 3, 0),
    ("d", "a", 0, 0),
    ("e", "b", 4, 1),
    ("f", "c", 4, 1),
    ("g", "d", 5, 1),
    ("a", "h", 5, 0),
    ("b", "e", 5, 1),
    ("c", "f", 6, 1),
    ("d", "g", 6, 0),
    ("e", "h", 6, 1),
]


def t_key(events, i):
    """Calkowity znacznik czasu zdarzenia i: (kubelek, pozycja na liscie)."""
    return (events[i][2], i)


def truncate(events, i):
    """Zdarzenia o czasie < t(e_i) — reszte usuwamy (zdarzenia 'przyszle' w usuwanych,
    o czasie >= t(e_i); zdarzenie e_i tez usuwamy — cechy e liczymy z pozostalych)."""
    thr = t_key(events, i)
    return [e for j, e in enumerate(events) if t_key(events, j) < thr]


def check(name, feats_full_fn):
    """feats_full_fn(events) -> tablica wierszy per zdarzenie (rownolegla do events)."""
    full = feats_full_fn(EVENTS)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(EVENTS), size=min(8, len(EVENTS)), replace=False):
        ev_tr = truncate(EVENTS, int(i))
        # cechy zdarzenia e_i w pelnym strumieniu musza rownac sie cechom 'e_i jako
        # nastepne zdarzenie' w strumieniu uciętym do t < t(e_i)
        partial = feats_full_fn(ev_tr + [EVENTS[i]])
        assert np.allclose(full[i], partial[-1], atol=1e-6), (
            f"{name}: wyciek w zdarzeniu {i}!\n  full:    {full[i]}\n"
            f"  partial: {partial[-1]}\n  cechy widza przyszlosc/przyszle zdarzenia"
        )
    RESULTS.append((name, "OK"))


# ================= 1. cascadebench: kontekst reczny =================
from cascadebench.features import (
    context as cb_context,
    hopper as cb_hopper,
    compa as cb_compa,
)
from cascadebench.graph import Graph

g = Graph(NBR)
g.index = {u: i for i, u in enumerate(NODES)}

check("cascadebench/context (kontekst reczny)", lambda ev: cb_context(g, ev, seed=0))
check("cascadebench/hopper (path-based)", lambda ev: cb_hopper(g, ev, seed=0))
check("cascadebench/compa (volume)", lambda ev: cb_compa(ev))

# ================= 2. framework-1: _features (kontekst reczny) =================
import exp_cascade as XC

sys.argv = [sys.argv[0]]  # wyczysc argv (SPLIT/SCALE_N)


def fw1_context(ev):
    recv_b, sent_b = XC._index_traffic(ev)
    _X1, Xc, _y = XC._features(ev, NBR, recv_b, sent_b, seed=0)
    return Xc


check("framework-1/_features (kontekst reczny)", fw1_context)


def fw1_comp(ev):
    from exp_cascade_sweep import _compa_features

    return _compa_features(ev)


check("framework-1/compa (volume)", fw1_comp)

# ================= 3. TGN-lite: pamiec per mini-batch =================
# Score zdarzenia z batcha i zalezy wylacznie od batchy < i (stan pamieci) —
# test: logity batcha i na pelnym strumieniu == logity batcha i gdy strumien
# obciety do batchy <= i (bez pozniejszych). Model z ustalonymi wagami.
from cascadebench.models import TemporalGNN, event_batches

torch.manual_seed(7)
n = len(NODES)
idx = {u: k for k, u in enumerate(NODES)}
Xn = torch.randn(n, 3)
ef = torch.randn(len(EVENTS), 4)
s_all = torch.tensor([idx[e[0]] for e in EVENTS])
v_all = torch.tensor([idx[e[1]] for e in EVENTS])
model = TemporalGNN(3, 4)
model.eval()
with torch.no_grad():
    full_logits = model(Xn, event_batches(g, EVENTS), n, ef, s_all, v_all)
    batches = event_batches(g, EVENTS)
    # obcinamy strumien do batcha nr k wlacznie (k = ostatni zachowany)
    for k in (1, 4, len(batches) - 1):
        kept = [i for _b, _sb, _vb, gidx in batches[: k + 1] for i in gidx]
        ev_tr = [EVENTS[i] for i in sorted(kept, key=lambda i: (EVENTS[i][2], i))]
        lg = model(Xn, event_batches(g, ev_tr), n, ef[kept], s_all[kept], v_all[kept])
        assert torch.allclose(lg, full_logits[kept], atol=1e-5), (
            f"TGN-lite: wyciek pamieci — logity batcha <= {k} zaleza od pozniejszych zdarzen"
        )
RESULTS.append(("cascadebench/TemporalGNN (pamiec per batch)", "OK"))

# ================= 4. TGAT-lite: jak wyzej =================
from cascadebench.models import TGATLite

torch.manual_seed(11)
model2 = TGATLite(3, 4)
model2.eval()
with torch.no_grad():
    full_logits2 = model2(Xn, event_batches(g, EVENTS), n, ef, s_all, v_all)
    for k in (1, 4, len(batches) - 1):
        kept = [i for _b, _sb, _vb, gidx in batches[: k + 1] for i in gidx]
        ev_tr = [EVENTS[i] for i in sorted(kept, key=lambda i: (EVENTS[i][2], i))]
        lg = model2(Xn, event_batches(g, ev_tr), n, ef[kept], s_all[kept], v_all[kept])
        assert torch.allclose(lg, full_logits2[kept], atol=1e-5), (
            f"TGAT-lite: wyciek pamieci — logity batcha <= {k} zaleza od pozniejszych zdarzen"
        )
RESULTS.append(("cascadebench/TGATLite (pamiec per batch)", "OK"))

# ================= podsumowanie =================
print("\n=== TESTY ANTY-WYCIEK (A1) ===")
for name, status in RESULTS:
    print(f"  [{status}] {name}")
print(
    f"\nLacznie: {len(RESULTS)} detektorow/modelow, wszystkie OK"
    if all(s == "OK" for _n, s in RESULTS)
    else "\nPORAZKA"
)
sys.exit(0 if all(s == "OK" for _n, s in RESULTS) else 1)
