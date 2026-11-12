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


def _time_order(events: List) -> List[int]:
    """Calkowity porzadek czasowy zdarzen (regula remisow): kubelek, potem pozycja na
    liscie strumienia. Zdarzenia w tym samym kubełku sa wspolczesne, ale dostaja ustalona
    kolejnosc tasowanej listy — losowa, deterministyczna dla danego przebiegu."""
    return sorted(range(len(events)), key=lambda i: (events[i][2], i))


def context(graph: Graph, events: List, seed: int) -> np.ndarray:
    """Cechy kontekstu kaskady. PRZYCZYNOWOSC ZDARZENIOWA (fix po recenzji wewnetrznej):
    zdarzenie e widzi wylacznie zdarzenia o scisle wczesniejszym znaczniku czasu t(e) =
    (kubelek, pozycja na liscie) — czyli wczesniejsze kubełki ORAZ wczesniejsze zdarzenia
    TEGO samego kubełka (w porzadku strumienia). Poprzednia wersja dzielila po kubełkach
    (<= b-1), przez co detektor odbiorczy nie widzial fali budujacej sie w biezacym
    kubełku — asymetria faworyzowala liczniki wolumetryczne pod blastem."""
    W = WINDOW
    n = len(events)
    X = [None] * n
    recv_b: dict = defaultdict(
        set
    )  # kubelek -> zbior odbiorcow (wypelniany przy przejsciu)
    sent_b: dict = defaultdict(set)  # kubelek -> zbior nadawcow
    for i in _time_order(events):
        s, v, b, _l = events[i]
        cs = graph.neighbors(v)
        deg_v = max(1, len(cs))
        f1 = [
            1.0 if s in cs else 0.0,
            1.0 if in_rhythm(graph, s, b, seed) else 0.0,
            float(len(cs)),
            float(len(graph.neighbors(s))),
        ]
        recent_recv = set(recv_b.get(b, ()))  # wczesniejsze zdarzenia TEGO kubełka
        recent_sent = set(sent_b.get(b, ()))
        for bb in range(max(0, b - W), b):  # kompletne wczesniejsze kubełki okna
            recent_recv |= recv_b.get(bb, set())
            recent_sent |= sent_b.get(bb, set())
        burst = len((cs - {s}) & recent_recv) / deg_v
        recency = 1.0 if s in recent_recv else 0.0
        prop = len((cs - {s}) & recent_sent) / deg_v
        X[i] = f1 + [burst, recency, prop]
        recv_b[b].add(v)
        sent_b[b].add(s)
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
    """Cechy wolumenowe COMPA. PRZYCZYNOWOSC NA POZIOMIE ZDARZENIA (fix po recenzji
    zewnetrznej): liczniki obejmuja wczesniejsze kubełki ORAZ wczesniejsze zdarzenia z
    biezacego kubełka (kolejnosc wewnatrz kubełka = ustalona kolejnosc na liscie zdarzen,
    czyli losowa ale stala — bezstronna). Podzial na poziomie kubełka ukrywal przed
    detektorem caly burst i czynil go slomiana kuke; kontekst odbiorczy i pamiec TGN
    pozostaja kubełkowo-przeszle [b-W, b-1]."""
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
            1.0 / (1.0 + pair_c.get((s, v), 0)),
            1.0 / (1.0 + sb_c.get((s, b), 0)),
            float(st_c.get(s, 0)),
        ]
        pair_c[(s, v)] = pair_c.get((s, v), 0) + 1
        sb_c[(s, b)] = sb_c.get((s, b), 0) + 1
        st_c[s] = st_c.get(s, 0) + 1
    return np.asarray(X, np.float32)


def hopper(graph: Graph, events: List, seed: int) -> np.ndarray:
    """Cechy w duchu Hoppera (Ho i in. 2021): detekcja \\emph{sciezki ruchu}. Zdarzenie podejrzane,
    gdy nadawca byl niedawno OSIAGNIETY (precursor przyczynowy --- czesc lancucha), RUSZA z~rozmachem
    do RZADKIEGO odbiorcy i~BEZ benign wyjasnienia (poza rytmem). Adaptacja do danych graf+czas ---
    bez tresci maila (URL/lure z~oryginalu sa niedostepne), wiec kodujemy strukturalno-czasowy rdzen
    sygnatury Hoppera.

    PRZYCZYNOWOSC ZDARZENIOWA (fix po recenzji wewnetrznej): precursor, rozmach (momentum) i
    popularnosc odbiorcy liczone ze zdarzen o scisle wczesniejszym t(e) = (kubelek, pozycja)
    — wczesniejsze kubełki okna [b-W, b-1] plus wczesniejsze zdarzenia TEGO kubełka.
    Poprzednio podzial po kubełkach (< b) ukrywal fale budujaca sie w biezacym kubełku."""
    W = WINDOW
    n = len(events)
    X = [None] * n
    recv_b: dict = defaultdict(set)  # kubelek -> odbiorcy (prefix)
    out_b: dict = defaultdict(set)  # (kubelek, nadawca) -> jego cele (prefix)
    dest_cnt: dict = defaultdict(
        int
    )  # popularnosc odbiorcy wsrod STRICTLY wczesniejszych
    for i in _time_order(events):
        s, v, b, _l = events[i]
        precursor = 0.0
        if s in recv_b.get(b, ()):
            precursor = 1.0
        else:
            for bb in range(max(0, b - W), b):
                if s in recv_b.get(bb, ()):
                    precursor = 1.0
                    break
        fan_set = set(out_b.get((b, s), ()))
        for bb in range(max(0, b - W), b):
            fan_set |= out_b.get((bb, s), set())
        fan = float(len(fan_set))
        rare_dest = 1.0 / (1.0 + dest_cnt[v])
        unexplained = 0.0 if in_rhythm(graph, s, b, seed) else 1.0
        hop_sig = precursor * unexplained
        X[i] = [precursor, fan, rare_dest, unexplained, hop_sig]
        recv_b[b].add(v)
        out_b[(b, s)].add(v)
        dest_cnt[v] += 1
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
