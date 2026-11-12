"""cascadebench.attack — atakujacy i scenariusze (komponowalne strategie ewazji).

Model zagrozenia: lateral phishing jako KASKADA propagacji. Strategia atakujacego rozdziela
pokretla ZASIEGU (fan-out K) od pokretel EWAZJI (rozproszenie w czasie g, mimikra rytmu, fabrykacja),
co umozliwia front Pareto i adaptacyjne ataki w przestrzeni problemu (problem-space, Pierazzi i in.).

Zdarzenie: krotka (nadawca, odbiorca, bucket, etykieta).  Scenario = ataki + leak-aware benign.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from .graph import Graph, rhythm_bucket

Event = Tuple[str, str, int, int]


@dataclass
class CascadeStrategy:
    """Strategia atakujacego. fanout = ZASIEG; spread/mimicry/fabrication = EWAZJA."""

    n_cascades: int = 30
    max_hops: int = 4
    p_infect: float = 0.6
    fanout: int = 8  # K — kontaktow zarazanych na nosiciela (zasieg)
    spread: int = 1  # g — odstep bucketow miedzy wyslaniami (ewazja czasowa)
    mimicry: float = 0.0  # frakcja wyslan w rytmie nadawcy (ewazja rytmu)
    fabrication: float = (
        0.0  # frakcja krawedzi fabrykowanych (placeholder/rozszerzenie)
    )
    timing: str = "det"  # "det": odstepy spread; "emp": odstepy z rozkladu tla

    def label(self) -> str:
        return (
            f"K{self.fanout}-g{self.spread}"
            + (f"-m{self.mimicry}" if self.mimicry else "")
            + (f"-f{self.fabrication}" if self.fabrication else "")
        )


@dataclass
class Scenario:
    graph: Graph
    events: List[Event]
    n_attacks: int
    infected_frac: float
    strategy: CascadeStrategy = field(default=None)


def cascade(
    graph: Graph, strat: CascadeStrategy, seed: int, gap_pool: List[int] = None
) -> Tuple[List[Event], int]:
    """Generuje zdarzenia ataku (etykieta 1) wg strategii. Zwraca (events, #zarazonych).

    R4-A1: timing="emp" — odstepy miedzy kolejnymi wyslaniami nosiciela sa losowane
    z empirycznego rozkladu odstepow tla (gap_pool, mierzony na benign tego harnessu);
    przy timing="det" zachowane deterministyczne odstepy spread (protokol jak dotad)."""
    rng = np.random.default_rng(seed)
    nodes = graph.nodes
    nb = graph.n_buckets
    events: List[Event] = []
    infected_all = set()
    seeds0 = [nodes[i] for i in rng.permutation(len(nodes))[: strat.n_cascades]]
    for ci, s0 in enumerate(seeds0):
        t0 = int(rng.integers(0, nb))
        infected = {s0}
        frontier = [(s0, t0)]
        for _hop in range(strat.max_hops):
            nxt = []
            for u, tu in frontier:
                real = sorted(graph.neighbors(u))
                rng.shuffle(real)
                real = [v for v in real if v not in infected]
                sent = 0
                attempts = 0
                while sent < strat.fanout and attempts < strat.fanout * 4 + 4:
                    attempts += 1
                    # EWAZJA przez fabrykacje: wyslanie do NIE-kontaktu (kontakt spoofowany OSINT,
                    # poza znanym grafem) zamiast do realnego sasiada.
                    if strat.fabrication and rng.random() < strat.fabrication:
                        v = nodes[int(rng.integers(len(nodes)))]
                        if v == u or v in infected or v in graph.neighbors(u):
                            continue
                    elif real:
                        v = real.pop()
                    elif strat.fabrication:
                        continue  # brak realnych — sprobuj fabrykacji w kolejnej iteracji
                    else:
                        break
                    bucket = (tu + 1 + sent * strat.spread) % nb
                    if strat.timing == "emp" and gap_pool:
                        # odstepy ataku z rozkladu tla: przesuniecie kumulatywne
                        # (tu + 1) + suma poprzednich odstepow z puli
                        extra = int(rng.integers(len(gap_pool)))
                        bucket = (tu + 1 + sent * (1 + gap_pool[extra])) % nb
                    if strat.mimicry and rng.random() < strat.mimicry:
                        bucket = rhythm_bucket(graph, u, f"{ci}:{u}:{v}", seed)
                    events.append((u, v, bucket, 1))
                    sent += 1
                    if rng.random() < strat.p_infect:
                        infected.add(v)
                        nxt.append((v, bucket))
            frontier = nxt
            if not frontier:
                break
        infected_all |= infected
    return events, len(infected_all)


def benign_traffic(
    graph: Graph, n_events: int, seed: int, off_hours: float = 0.0, matched: bool = True
) -> List[Event]:
    """Ruch legalny (etykieta 0). matched=True: losowe krawedzie kontaktu (rozklad jak atak) —
    usuwa konfundent stopnia/tozsamosci. off_hours: frakcja poza rytmem (leak-aware)."""
    rng = np.random.default_rng(seed + 777)
    nb = graph.n_buckets
    edges = graph.edges()
    out: List[Event] = []
    for _ in range(n_events):
        u, w = edges[int(rng.integers(len(edges)))]
        b = int(rng.integers(nb))
        # off-hours: cz prob off_hours benign POZA rytmem nadawcy (kontrola wycieku rytmu)
        if off_hours and rng.random() < off_hours:
            act = graph.active_buckets(u)
            if act:
                off = sorted(set(range(nb)) - act) or [b]
                b = off[int(rng.integers(len(off)))]
        out.append((u, w, b, 0))
    return out


def _benign_gap_pool(
    graph: Graph, n_events: int, seed: int, off_hours: float
) -> List[int]:
    """Empiryczny rozklad odstepow tla: odstepy (kubelki) miedzy kolejnymi wysylaniami
    tego samego nadawcy w benign (protokol tego harnessu, z off_hours)."""
    rng = np.random.default_rng(seed + 555)
    nb = graph.n_buckets
    edges = graph.edges()
    by_sender: Dict[Any, List[int]] = {}
    for _ in range(n_events):
        u, w = edges[int(rng.integers(len(edges)))]
        b = int(rng.integers(nb))
        if off_hours and rng.random() < off_hours:
            act = graph.active_buckets(u)
            if act:
                off = sorted(set(range(nb)) - act) or [b]
                b = off[int(rng.integers(len(off)))]
        by_sender.setdefault(u, []).append(b)
    pool: List[int] = []
    for bs in by_sender.values():
        bs.sort()
        pool.extend(b2 - b1 for b1, b2 in zip(bs, bs[1:]) if b2 > b1)
    return pool or [1]


def build_scenario(
    graph: Graph,
    strat: CascadeStrategy,
    seed: int,
    benign_ratio: int = 3,
    off_hours: float = 0.0,
) -> Scenario:
    """Pelny scenariusz: ataki (kaskada) + leak-aware benign, przetasowane.

    R4-A1: przy strat.timing=="emp" najpierw mierzy rozklad odstepow tla
    (pilot benign w tym samym protokole), potem generuje atak z odstepami
    z tego rozkladu (kolejnosc: pilot -> atak -> wlasciwy benign)."""
    if strat.timing == "emp":
        pool = _benign_gap_pool(graph, 3000, seed, off_hours)
    else:
        pool = None
    atk, n_inf = cascade(graph, strat, seed, gap_pool=pool)
    ben = benign_traffic(
        graph, benign_ratio * max(1, len(atk)), seed, off_hours=off_hours
    )
    ev = atk + ben
    np.random.default_rng(seed + 13).shuffle(ev)
    return Scenario(graph, ev, len(atk), n_inf / max(1, len(graph)), strat)
