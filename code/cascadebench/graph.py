"""cascadebench.graph — abstrakcja grafu komunikacji (syntetyk + realne topologie).

Jedna klasa `Graph` ujednolica:
  * syntetyk proceduralny (organizacje, dowolne N) — kontrolowalny, bez LLM,
  * realne topologie z list krawedzi ze znacznikami czasu (SNAP: email-Eu-core, CollegeMsg),
  * realny graf Enrona z naglowkow (opcjonalnie).
Graf udostepnia sasiedztwo i — gdy dostepny — REALNY rytm aktywnosci nadawcow (z czasu).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent

N_BUCKETS = 28                      # 7 dni x 4 bloki 6h (rytm tygodniowy)
_BUCKET_SECONDS = 21600


class Graph:
    """Graf kontaktow. `nbr`: wezel -> zbior kontaktow. `rhythm`: wezel -> aktywne buckety (opcjonalnie)."""

    def __init__(self, nbr: dict, rhythm: Optional[dict] = None, name: str = "graph",
                 n_buckets: int = N_BUCKETS):
        self.nbr = {u: set(vs) for u, vs in nbr.items() if vs}
        self.rhythm = rhythm or {}
        self.name = name
        self.n_buckets = n_buckets
        self.nodes = sorted(self.nbr)
        self.index = {u: i for i, u in enumerate(self.nodes)}

    def __len__(self) -> int:
        return len(self.nodes)

    def neighbors(self, u) -> set:
        return self.nbr.get(u, set())

    def edges(self) -> list:
        return [(u, w) for u in self.nbr for w in self.nbr[u]]

    def active_buckets(self, u) -> Optional[set]:
        return self.rhythm.get(u)

    # ----------------------------- konstruktory ------------------------------
    @classmethod
    def synthetic(cls, n_twins: int = 1600) -> "Graph":
        """Proceduralny graf organizacji (kontrolowalny, dowolne N, bez LLM)."""
        import sys
        sys.path.insert(0, str(_CODE))
        from data.org_graph import build_structural, contacts
        nodes = build_structural(n_twins)
        nbr = defaultdict(set)
        for tid, node in nodes.items():
            for c in contacts(node):
                nbr[tid].add(c); nbr[c].add(tid)
        return cls(nbr, name=f"synthetic-{n_twins}")

    @classmethod
    def from_edgelist(cls, path, min_deg: int = 2, core_cap: int = 2000,
                      name: Optional[str] = None) -> "Graph":
        """Realna topologia z listy krawedzi 'src dst [timestamp]'. Wyprowadza realny rytm,
        gdy obecny timestamp. Przycina do aktywnego rdzenia (stopien >= min_deg, top-K)."""
        path = Path(path)
        nbr_full = defaultdict(set)
        sent = defaultdict(list)
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) < 2:
                continue
            u, v = p[0], p[1]
            if u == v:
                continue
            nbr_full[u].add(v); nbr_full[v].add(u)
            if len(p) >= 3:
                try:
                    sent[u].append((int(p[2]) // _BUCKET_SECONDS) % N_BUCKETS)
                except ValueError:
                    pass
        deg = {a: len(s) for a, s in nbr_full.items() if len(s) >= min_deg}
        core = set(sorted(deg, key=deg.get, reverse=True)[:core_cap])
        nbr = {a: (nbr_full[a] & core) for a in core}
        rhythm = {}
        for a in core:
            if sent.get(a):
                cnt = Counter(sent[a])
                rhythm[a] = {b for b, c in cnt.items() if c >= 2}
        return cls(nbr, rhythm, name=name or path.stem)

    @classmethod
    def enron(cls, max_users: int = 40, core_cap: int = 2000) -> "Graph":
        """Realny graf kontaktow Enrona z naglowkow From->To + rytm z Date."""
        import sys
        sys.path.insert(0, str(_CODE / "experiments"))
        import exp_enron_multiplex as EM
        EM.MAX_USERS = max_users
        EM.ENRON = _CODE.parents[1] / "projects/personalized-phishing-defense/code/data/enron/maildir"
        sent, active, contact, _corecip, _domain = EM.parse()
        nbr_full = defaultdict(set)
        for s, v in sent.items():
            for r, _b in v:
                nbr_full[s].add(r); nbr_full[r].add(s)
        deg = {a: len(s) for a, s in nbr_full.items() if len(s) >= 2}
        core = set(sorted(deg, key=deg.get, reverse=True)[:core_cap])
        nbr = {a: (nbr_full[a] & core) for a in core}
        rhythm = {a: active[a] for a in core if a in active}
        return cls(nbr, rhythm, name="enron")


def _synth_rhythm():
    import sys
    sys.path.insert(0, str(_CODE))
    from graph.build_temporal_overlay import is_consistent, in_bucket
    return is_consistent, in_bucket


def in_rhythm(graph: "Graph", node, bucket: int, seed: int) -> bool:
    """Czy `node` jest w rytmie w `bucket`. Realny rytm jesli dostepny, inaczej syntetyczny."""
    act = graph.active_buckets(node)
    if act is not None:
        return bucket in act
    is_consistent, _ = _synth_rhythm()
    return bool(is_consistent(node, bucket, seed))


def rhythm_bucket(graph: "Graph", node, salt: str, seed: int) -> int:
    """Zwraca bucket W rytmie `node` (do mimikry). Realny rytm jesli dostepny, inaczej syntetyczny."""
    act = graph.active_buckets(node)
    if act:
        a = sorted(act)
        return a[int.from_bytes(salt.encode()[:4] or b"\0", "little") % len(a)]
    _, in_bucket = _synth_rhythm()
    return int(in_bucket(node, salt, seed))


# rejestr standardowych grafow poligonu
REAL_GRAPHS = {
    "email-Eu-core": _CODE / "data" / "realgraphs" / "eu.txt",
    "CollegeMsg": _CODE / "data" / "realgraphs" / "college.txt",
    "sx-mathoverflow": _CODE / "data" / "realgraphs" / "sx-mathoverflow.txt",
    "sx-askubuntu": _CODE / "data" / "realgraphs" / "sx-askubuntu.txt",
    "sx-superuser": _CODE / "data" / "realgraphs" / "sx-superuser.txt",
}


def load(spec: str) -> Graph:
    """Fabryka: 'synthetic[:N]' | 'enron' | nazwa-realnego-grafu | sciezka do edgelist."""
    if spec.startswith("synthetic"):
        n = int(spec.split(":")[1]) if ":" in spec else 1600
        return Graph.synthetic(n)
    if spec == "enron":
        return Graph.enron()
    if spec in REAL_GRAPHS:
        return Graph.from_edgelist(REAL_GRAPHS[spec], name=spec)
    return Graph.from_edgelist(spec)
