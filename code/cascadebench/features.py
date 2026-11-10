"""cascadebench.features — ekstrakcja cech ze strumienia zdarzen (parametryzowana grafem).

Wspolne dla detektorow: cechy 1-hop, kontekst kaskady (burst/recency/propagacja), cechy wolumenowe
(COMPA), cechy strukturalne wezla. Rytm liczony przez graph.in_rhythm (realny lub syntetyczny).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import List

import numpy as np

from .graph import Graph, in_rhythm

WINDOW = 2


def index_traffic(events):
    recv = defaultdict(set)
    sent = defaultdict(set)
    for s, v, b, _l in events:
        recv[b].add(v)
        sent[b].add(s)
    return recv, sent


def onehop(graph: Graph, events: List, seed: int) -> np.ndarray:
    X = []
    for s, v, b, _l in events:
        cs = graph.neighbors(v)
        X.append(
            [
                1.0 if s in cs else 0.0,
                1.0 if in_rhythm(graph, s, b, seed) else 0.0,
                float(len(cs)),
                float(len(graph.neighbors(s))),
            ]
        )
    return np.asarray(X, np.float32)


def context(graph: Graph, events: List, seed: int) -> np.ndarray:
    """Cechy kontekstu kaskady. ZASADA LEAK-AWARE: kazda cecha strumieniowa liczona wylacznie
    z kubelkow PRZESZLYCH (<= b-1), zgodnie z pamiecia TGN (stan sprzed bucketa b)."""
    recv, sent = index_traffic(events)
    X = []
    for s, v, b, _l in events:
        cs = graph.neighbors(v)
        deg_v = max(1, len(cs))
        f1 = [
            1.0 if s in cs else 0.0,
            1.0 if in_rhythm(graph, s, b, seed) else 0.0,
            float(len(cs)),
            float(len(graph.neighbors(s))),
        ]
        recent = set()
        for bb in range(max(0, b - WINDOW), b):
            recent |= recv.get(bb, set())
        burst = len((cs - {s}) & recent) / deg_v
        recency = 0.0
        for bb in range(max(0, b - WINDOW), b):
            if s in recv.get(bb, set()):
                recency = 1.0
                break
        prop = (
            len(
                (cs - {s})
                & {
                    x
                    for bb in range(max(0, b - WINDOW), b)
                    for x in sent.get(bb, set())
                }
            )
            / deg_v
        )
        X.append(f1 + [burst, recency, prop])
    return np.asarray(X, np.float32)


def _past_counters(events: List) -> dict:
    """Liczniki wylacznie przeszle (bucket < b). Zwraca b -> (pair, sb, st):
    slowniki licznikow zdarzen z kubelkow < b (stan obserwowany PRZED kubelkiem b)."""
    counts = {}
    for s, v, b, _l in events:
        c = counts.setdefault(
            b,
            {"pair": defaultdict(int), "sb": defaultdict(int), "st": defaultdict(int)},
        )
        c["pair"][(s, v)] += 1
        c["sb"][(s, b)] += 1
        c["st"][s] += 1
    cum_p, cum_sb, cum_st = defaultdict(int), defaultdict(int), defaultdict(int)
    past = {}
    for b in sorted(counts):
        past[b] = (
            dict(cum_p),
            dict(cum_sb),
            dict(cum_st),
        )  # snapshot przed kubelkiem b
        c = counts[b]
        for k, x in c["pair"].items():
            cum_p[k] += x
        for k, x in c["sb"].items():
            cum_sb[k] += x
        for k, x in c["st"].items():
            cum_st[k] += x
    return past


def compa(events: List) -> np.ndarray:
    """Cechy wolumenowe COMPA. Liczniki (para, nadawca-bucket, nadawca) liczone wylacznie
    z kubelkow PRZESZLYCH (< b) — bez biezacego bucketa i bez przyszlosci (leak-aware)."""
    past = _past_counters(events)
    X = []
    for s, v, b, _l in events:
        p, sb, st = past.get(b, ({}, {}, {}))
        X.append(
            [
                1.0 / (1.0 + p.get((s, v), 0)),
                1.0 / (1.0 + sb.get((s, b), 0)),
                float(st.get(s, 0)),
            ]
        )
    return np.asarray(X, np.float32)


def hopper(graph: Graph, events: List, seed: int) -> np.ndarray:
    """Cechy w duchu Hoppera (Ho i in. 2021): detekcja \\emph{sciezki ruchu}. Zdarzenie podejrzane,
    gdy nadawca byl niedawno OSIAGNIETY (precursor przyczynowy --- czesc lancucha), RUSZA z~rozmachem
    do RZADKIEGO odbiorcy i~BEZ benign wyjasnienia (poza rytmem). Adaptacja do danych graf+czas ---
    bez tresci maila (URL/lure z~oryginalu sa niedostepne), wiec kodujemy strukturalno-czasowy rdzen
    sygnatury Hoppera.

    LEAK-AWARE (fix po recenzji): rozmach ruchu (momentum) liczony z kubelkow PRZESZLYCH
    [b-W, b-1] (poprzednia definicja [b, b+W] zawierala przyszle kubelki = look-ahead),
    a popularnosc odbiorcy z ruchu PRZED kubelkiem b."""
    recv, sent = index_traffic(events)
    dest_past = {}  # b -> Counter popularnosci odbiorcow z kubelkow < b
    cnt = {}
    for _s, v, b, _l in events:
        cnt.setdefault(b, Counter())[v] += 1
    cum = Counter()
    for b in sorted(cnt):
        dest_past[b] = cum.copy()  # snapshot przed kubelkiem b
        cum.update(cnt[b])
    out_win = defaultdict(set)  # (s,b) -> zbior celow w oknie PRZESZLYM [b-W, b-1]
    for s, v, b, _l in events:
        for bb in range(
            b + 1, b + WINDOW + 1
        ):  # cel liczony do okna od kubelka bb-W do bb-1
            out_win[(s, bb)].add(v)
    X = []
    for s, v, b, _l in events:
        precursor = 0.0  # czy s otrzymal cos w [b-W, b-1] (osiagniety)
        for bb in range(max(0, b - WINDOW), b):
            if s in recv.get(bb, set()):
                precursor = 1.0
                break
        fan = float(len(out_win.get((s, b), ())))  # rozmach ruchu w oknie PRZESZLYM
        rare_dest = 1.0 / (
            1.0 + dest_past.get(b, Counter())[v]
        )  # rzadki odbiorca (ruch przeszly)
        unexplained = (
            0.0 if in_rhythm(graph, s, b, seed) else 1.0
        )  # brak benign wyjasnienia (poza rytmem)
        hop_sig = precursor * unexplained  # sygnatura: ruch w lancuchu bez wyjasnienia
        X.append([precursor, fan, rare_dest, unexplained, hop_sig])
    return np.asarray(X, np.float32)


def node_features(graph: Graph, events: List, train_mask=None) -> np.ndarray:
    """Statyczne cechy wezla: stopien grafu + in/out-degree ze strumienia. LEAK-AWARE (fix po
    recenzji): gdy podano train_mask, stopnie strumieniowe liczone wylacznie ze zdarzen
    TRENINGOWYCH (poprzednio caly strumien = wyciek zbioru testowego i przyszlych zdarzen)."""
    import torch

    n = len(graph)
    idx = graph.index
    stream = (
        events
        if train_mask is None
        else [e for e, keep in zip(events, train_mask) if keep]
    )
    indeg = Counter(v for _s, v, _b, _l in stream)
    outdeg = Counter(s for s, _v, _b, _l in stream)
    X = torch.zeros(n, 3)
    for u, i in idx.items():
        X[i, 0] = len(graph.neighbors(u))
        X[i, 1] = indeg.get(u, 0)
        X[i, 2] = outdeg.get(u, 0)
    X = torch.log1p(X)
    return (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp(min=1e-6)
