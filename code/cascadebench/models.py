"""cascadebench.models — lekkie modele grafowe (czysty torch, bez PyG).

StaticGNN: agregacja po (statycznym) grafie kontaktow — widzi topologie, nie czas.
TemporalGNN (TGN-lite): pamiec wezla aktualizowana bucket-po-buckecie z obserwowalnego ruchu;
  zdarzenie klasyfikowane stanem SPRZED aktualizacji bucketa (brak wycieku).
"""
from __future__ import annotations

import torch
import torch.nn as nn

DIM = 24


class StaticGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, dim: int = DIM):
        super().__init__()
        self.proj = nn.Linear(node_dim, dim)
        self.w1 = nn.Linear(dim, dim); self.w2 = nn.Linear(dim, dim)
        self.clf = nn.Sequential(nn.Linear(dim * 3 + edge_dim, 48), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(48, 1))

    def forward(self, A, Xn, s_idx, v_idx, edge_feat):
        h = torch.relu(self.proj(Xn))
        h = torch.relu(A @ self.w1(h)); h = A @ self.w2(h)
        e = torch.cat([h[s_idx], h[v_idx], h[s_idx] * h[v_idx], edge_feat], dim=1)
        return self.clf(e).squeeze(1)


class TemporalGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, dim: int = DIM):
        super().__init__()
        self.proj = nn.Linear(node_dim, dim)
        self.msg = nn.Linear(dim, dim)
        self.upd = nn.GRUCell(dim, dim)
        self.clf = nn.Sequential(nn.Linear(dim * 3 + edge_dim, 48), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(48, 1))

    def forward(self, Xn, batches, n, edge_feat, s_all, v_all):
        M = torch.relu(self.proj(Xn))
        logits = torch.zeros(len(s_all))
        for _b, sb, vb, gidx in batches:
            if len(gidx) == 0:
                continue
            e = torch.cat([M[s_all[gidx]], M[v_all[gidx]],
                           M[s_all[gidx]] * M[v_all[gidx]], edge_feat[gidx]], dim=1)
            logits = logits.clone(); logits[gidx] = self.clf(e).squeeze(1)
            msg = torch.relu(self.msg(M[sb]))
            agg = torch.zeros(n, msg.shape[1]).index_add_(0, vb, msg)
            cnt = torch.zeros(n).index_add_(0, vb, torch.ones(len(vb))).clamp(min=1).unsqueeze(1)
            rec = torch.unique(vb)
            M = M.clone(); M[rec] = self.upd((agg / cnt)[rec], M[rec])
        return logits


class TGATLite(nn.Module):
    """Temporalny GNN z UWAGA (segment-softmax) + kodowanie czasu — druga, niezalezna architektura."""
    def __init__(self, node_dim: int, edge_dim: int, dim: int = DIM, tdim: int = 8):
        super().__init__()
        import math
        self._sqrt = math.sqrt(dim)
        self.proj = nn.Linear(node_dim, dim)
        self.qry = nn.Linear(dim, dim); self.key = nn.Linear(dim + tdim, dim); self.val = nn.Linear(dim, dim)
        self.upd = nn.GRUCell(dim, dim)
        self.freq = nn.Parameter(torch.randn(tdim) * 0.1)
        self.clf = nn.Sequential(nn.Linear(dim * 3 + edge_dim, 48), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(48, 1))

    def forward(self, Xn, batches, n, edge_feat, s_all, v_all):
        M = torch.relu(self.proj(Xn))
        logits = torch.zeros(len(s_all))
        for _i, (b, sb, vb, gidx) in enumerate(batches):
            if len(gidx) == 0:
                continue
            e = torch.cat([M[s_all[gidx]], M[v_all[gidx]],
                           M[s_all[gidx]] * M[v_all[gidx]], edge_feat[gidx]], dim=1)
            logits = logits.clone(); logits[gidx] = self.clf(e).squeeze(1)
            te = torch.cos(torch.full((len(sb), 1), float(b)) * self.freq.unsqueeze(0))
            q = self.qry(M[vb]); k = self.key(torch.cat([M[sb], te], dim=1)); val = self.val(M[sb])
            score = (q * k).sum(1) / self._sqrt
            score = score - score.max(); ex = score.exp()
            denom = torch.zeros(n).index_add(0, vb, ex)
            att = ex / denom[vb].clamp(min=1e-9)
            agg = torch.zeros(n, val.shape[1]).index_add(0, vb, att.unsqueeze(1) * val)
            rec = torch.unique(vb)
            M = M.clone(); M[rec] = self.upd(agg[rec], M[rec])
        return logits


def norm_adj(graph, n: int) -> torch.Tensor:
    A = torch.eye(n); idx = graph.index
    for u, nbs in graph.nbr.items():
        for w in nbs:
            if u in idx and w in idx:
                A[idx[u], idx[w]] = 1.0
    return A / A.sum(1, keepdim=True).clamp(min=1.0)


CHUNK = 64  # wielkosc mini-batchu zdarzen (regula remisow: batch jest wspolczesny)


def event_batches(graph, events, chunk=CHUNK):
    """Mini-batche zdarzen w porzadku calkowitym t(e) = (kubelek, pozycja na liscie).
    Batch nie przekracza granicy kubełka; zdarzenia w jednym batchu sa WSPOLCZESNE
    (regula remisow: nie widza sie nawzajem — rozbicie tasowanej listy na kawalki po
    <= chunk). Batch i widzi stan pamieci wylacznie po batchach < i (przyczynowosc
    zdarzeniowa zamiast kubełkowej). Zwraca liste (bucket, s_idx, v_idx, gidx)."""
    idx = graph.index
    by_b = {}
    for i, e in enumerate(events):
        by_b.setdefault(e[2], []).append(i)
    out = []
    for b in sorted(by_b):
        g = by_b[b]
        for c in range(0, len(g), chunk):
            part = g[c:c + chunk]
            sb = torch.tensor([idx[events[i][0]] for i in part])
            vb = torch.tensor([idx[events[i][1]] for i in part])
            out.append((b, sb, vb, part))
    return out


def per_bucket(graph, events):
    idx = graph.index; pb = []
    for b in range(graph.n_buckets):
        gi = [i for i, e in enumerate(events) if e[2] == b]
        if not gi:
            pb.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), []))
            continue
        pb.append((torch.tensor([idx[events[i][0]] for i in gi]),
                   torch.tensor([idx[events[i][1]] for i in gi]), gi))
    return pb
