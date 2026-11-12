"""Mocniejszy baseline uczony: TGAT-lite (uwaga + kodowanie czasu) vs TGN-lite (GRU) vs reczny (#GP-EXP-22).

Odpowiada na zarzut 'wygrywacie tylko z cechami recznymi / slabym GNN': pokazuje, ze przewaga
temporalna NIE zalezy od konkretnej architektury. Dwa NIEZALEZNE uczone modele temporalne:
  * TGN-lite (GRU)  -- pamiec wezla usredniana, jak w exp_cascade
  * TGAT-lite       -- pamiec aktualizowana ATENCYJNIE (segment-softmax) z kodowaniem czasu (Bochner)
Oba porownane z recznym kontekstem (tab_ctx) na syntetycznej kaskadzie (split po ofiarach, 5 ziaren).

Wyjscie: results/exp_cascade_arch.csv
"""
from __future__ import annotations

import csv
import math
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

from exp_cascade import (T_BUCKETS, _simulate, _features, _index_traffic,        # noqa: E402
                         _node_features, _recall_at_fpr, TemporalGNN)

SEEDS = list(range(5))
DIM = 24
EPOCHS = 120


class TGATLite(nn.Module):
    """Temporalny GNN z UWAGA: pamiec wezla aktualizowana atencyjnie (segment-softmax) po
    komunikatach w buckecie, z funkcyjnym kodowaniem czasu. Niezalezna architektura wzgledem TGN-lite."""
    def __init__(self, node_dim, edge_dim, dim=DIM, tdim=8):
        super().__init__()
        self.proj = nn.Linear(node_dim, dim)
        self.qry = nn.Linear(dim, dim)
        self.key = nn.Linear(dim + tdim, dim)
        self.val = nn.Linear(dim, dim)
        self.upd = nn.GRUCell(dim, dim)
        self.freq = nn.Parameter(torch.randn(tdim) * 0.1)
        self.clf = nn.Sequential(nn.Linear(dim * 3 + edge_dim, 48), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(48, 1))

    def _tenc(self, b, E):
        dt = torch.full((E, 1), float(b))
        return torch.cos(dt * self.freq.unsqueeze(0))               # [E, tdim]

    def forward(self, Xn, batches, n, edge_feat, s_all, v_all):
        M = torch.relu(self.proj(Xn))
        logits = torch.zeros(len(s_all))
        for _i, (b, sb, vb, gidx) in enumerate(batches):
            if len(gidx) == 0:
                continue
            e = torch.cat([M[s_all[gidx]], M[v_all[gidx]],
                           M[s_all[gidx]] * M[v_all[gidx]], edge_feat[gidx]], dim=1)
            logits = logits.clone(); logits[gidx] = self.clf(e).squeeze(1)
            q = self.qry(M[vb])
            k = self.key(torch.cat([M[sb], self._tenc(b, len(sb))], dim=1))
            val = self.val(M[sb])
            score = (q * k).sum(1) / math.sqrt(q.shape[1])
            score = score - score.max()                              # stabilnosc (stala globalna)
            ex = score.exp()
            denom = torch.zeros(n).index_add(0, vb, ex)
            att = ex / denom[vb].clamp(min=1e-9)                      # segment-softmax po odbiorcach
            agg = torch.zeros(n, val.shape[1]).index_add(0, vb, att.unsqueeze(1) * val)
            rec = torch.unique(vb)
            M = M.clone(); M[rec] = self.upd(agg[rec], M[rec])
        return logits


def _per_bucket(events, idx, chunk=64):
    """Mini-batche zdarzen w porzadku (kubelek, pozycja) — przyczynowosc zdarzeniowa
    (fix po recenzji wewnetrznej): batch nie przekracza kubełka, batch jest wspolczesny."""
    by_b = {}
    for i, e in enumerate(events):
        by_b.setdefault(e[2], []).append(i)
    pb = []
    for b in sorted(by_b):
        g = by_b[b]
        for c in range(0, len(g), chunk):
            part = g[c:c + chunk]
            sb = torch.tensor([idx[events[i][0]] for i in part])
            vb = torch.tensor([idx[events[i][1]] for i in part])
            pb.append((b, sb, vb, part))
    return pb


def _train(model, Xn, pb, n, ef, s_idx, v_idx, yt, tr_t, te_t, y, te):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xn, pb, n, ef, s_idx, v_idx)[tr_t], yt[tr_t])
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        s = torch.sigmoid(model(Xn, pb, n, ef, s_idx, v_idx))[te_t].numpy()
    return roc_auc_score(y[te], s), _recall_at_fpr(y[te], s)


def run_seed(seed):
    torch.manual_seed(seed)
    twins, nbr, events = _simulate(seed)
    recv_b, sent_b = _index_traffic(events)
    X1, Xc, _yf = _features(events, nbr, recv_b, sent_b, seed)
    idx = {t: i for i, t in enumerate(twins)}; n = len(twins)
    y = np.array([e[3] for e in events], dtype=np.float32)
    vics = sorted({e[1] for e in events})
    rng = np.random.default_rng(seed)
    te_v = set(np.array(vics)[rng.permutation(len(vics))[:max(1, len(vics) // 3)]])
    te = np.array([e[1] in te_v for e in events]); tr = ~te

    out = {}
    cc = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(Xc[tr], y[tr])
    sc = cc.predict_proba(Xc[te])[:, 1]
    out["tab_ctx"] = (roc_auc_score(y[te], sc), _recall_at_fpr(y[te], sc))

    Xn = _node_features(events, nbr, idx, n, recv_b, sent_b, tr=tr)
    s_idx = torch.tensor([idx[e[0]] for e in events])
    v_idx = torch.tensor([idx[e[1]] for e in events])
    ef = torch.tensor(X1, dtype=torch.float32)
    yt = torch.tensor(y); tr_t = torch.tensor(tr); te_t = torch.tensor(te)
    pb = _per_bucket(events, idx)
    out["gnn_gru"] = _train(TemporalGNN(Xn.shape[1], X1.shape[1]), Xn, pb, n, ef,
                            s_idx, v_idx, yt, tr_t, te_t, y, te)
    out["gnn_attn"] = _train(TGATLite(Xn.shape[1], X1.shape[1]), Xn, pb, n, ef,
                             s_idx, v_idx, yt, tr_t, te_t, y, te)
    return out


def main():
    agg = {}
    for seed in SEEDS:
        res = run_seed(seed)
        for k, (a, r) in res.items():
            agg.setdefault(k, {"auc": [], "r1": []})
            agg[k]["auc"].append(a); agg[k]["r1"].append(r)
        print("  seed=%d: " % seed + " | ".join(f"{k} auc={v[0]:.3f} r1={v[1]:.3f}"
                                                 for k, v in res.items()), flush=True)
    out = RESULTS / "exp_cascade_arch.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model", "auc", "recall_fpr1", "n_seeds"])
        for k, v in agg.items():
            w.writerow([k, round(float(np.mean(v["auc"])), 4),
                        round(float(np.mean(v["r1"])), 4), len(v["auc"])])
    print(f"\n[arch] -> {out}")
    for k in ["tab_ctx", "gnn_gru", "gnn_attn"]:
        if k in agg:
            print(f"  {k}: AUC={np.mean(agg[k]['auc']):.3f}  R@FPR1%={np.mean(agg[k]['r1']):.3f}")


if __name__ == "__main__":
    main()
