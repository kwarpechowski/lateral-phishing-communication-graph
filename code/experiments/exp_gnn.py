"""Multipleksowy GNN (czysty torch, bez PyG) vs reczne cechy multipleksu (#GP-EXP-8).

Pytanie: czy UCZONA reprezentacja grafowa bije reczne cechy (per-warstwa membership +
pokrycie + czas) z LightGBM? GNN: uczone osadzenia wezlow -> 2 warstwy agregacji sredniej
PER warstwa multipleksu (contact + OSINT + temporal-coactivity) -> konkatenacja (fuzja
multipleksowa) -> klasyfikacja krawedzi (nadawca,ofiara) + cecha sp. czasowej. Trenowany
end-to-end. Porownanie AUC + Recall@FPR=1% z baseline LightGBM. 5 ziaren, split po ofiarach.

Wyjscie: results/exp_gnn.csv
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from graph.build_layers import build_layers                          # noqa: E402
from graph.build_osint_overlay import build_overlay                  # noqa: E402
from graph.build_temporal_overlay import (coactivity_edges, in_bucket,  # noqa: E402
                                          is_consistent, off_bucket)

SEEDS = list(range(5))
EVENTS_PER_VICTIM = 24
DIM = 32
EPOCHS = 150
torch.manual_seed(0)


def _h(*p):
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def _recall_at_fpr(y, s, t=0.01):
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(y, s)
    m = fpr <= t
    return float(tpr[m].max()) if m.any() else 0.0


def _norm_adj(edges, idx, n):
    A = torch.eye(n)
    for a, b in edges:
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            A[i, j] = 1.0; A[j, i] = 1.0
    d = A.sum(1, keepdim=True).clamp(min=1.0)
    return A / d


class MultiplexGNN(nn.Module):
    def __init__(self, n_nodes, n_layers, dim=DIM):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, dim)
        self.w1 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.w2 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.clf = nn.Sequential(
            nn.Linear(dim * n_layers * 3 + 1, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1))

    def node_reps(self, adjs):
        X = self.emb.weight
        reps = []
        for A, w1, w2 in zip(adjs, self.w1, self.w2):
            h = torch.relu(A @ w1(X))
            h = A @ w2(h)
            reps.append(h)
        return torch.cat(reps, dim=1)                       # [n, dim*n_layers]

    def forward(self, adjs, s_idx, v_idx, tfeat):
        H = self.node_reps(adjs)
        hs, hv = H[s_idx], H[v_idx]
        e = torch.cat([hs, hv, hs * hv, tfeat.unsqueeze(1)], dim=1)
        return self.clf(e).squeeze(1)


def _self_rows():
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _events_and_graph(seed):
    base = build_layers()
    ov, _ = build_overlay(seed=seed, p_cross=0.5, fabrication_rate=0.0)
    twins_all = sorted({r["twin_id"] for r in _self_rows()})
    layers = {"contact": base["contact"], **ov,
              "temporal_coactivity": coactivity_edges(twins_all, seed)}
    names = list(layers)
    # adjacency dicts (dla generacji zdarzen) + znormalizowane macierze (dla GNN)
    adjd = {}
    for nm, edges in layers.items():
        nbr = {}
        for e in edges:
            nbr.setdefault(e[0], set()).add(e[1]); nbr.setdefault(e[1], set()).add(e[0])
        adjd[nm] = nbr
    contact_nbr = adjd["contact"]
    osint_union = {}
    for L in names[1:-1]:
        for t, nb in adjd[L].items():
            osint_union.setdefault(t, set()).update(nb)
    rows = []
    for v in twins_all:
        c = sorted(contact_nbr.get(v, set()))
        o_only = sorted(osint_union.get(v, set()) - contact_nbr.get(v, set()) - {v})
        off = [t for t in twins_all if t != v and t not in contact_nbr.get(v, set())
               and t not in osint_union.get(v, set())]
        if not c or not off:
            continue
        for k in range(EVENTS_PER_VICTIM):
            h = _h(str(seed), v, str(k)); salt = f"{v}:{k}"
            if h % 2 == 0:
                r = (h // 3) % 100
                if r < 40:
                    s = c[h % len(c)]
                elif r < 70 and o_only:
                    s = o_only[h % len(o_only)]
                else:
                    s = off[h % len(off)]
                b = off_bucket(s, salt, seed) if (h // 7) % 100 < 15 else in_bucket(s, salt, seed)
                rows.append((s, v, 0, b))
            else:
                mimic = (h // 5) % 100 < 25
                s = off[h % len(off)] if (h // 11) % 2 == 0 else c[h % len(c)]
                b = in_bucket(s, salt, seed) if mimic else off_bucket(s, salt, seed)
                rows.append((s, v, 1, b))
    return twins_all, layers, names, rows


def run(seed):
    twins, layers, names, rows = _events_and_graph(seed)
    idx = {t: i for i, t in enumerate(twins)}
    adjs = [_norm_adj(layers[nm], idx, len(twins)) for nm in names]

    s_idx = torch.tensor([idx[r[0]] for r in rows])
    v_idx = torch.tensor([idx[r[1]] for r in rows])
    y = np.array([r[2] for r in rows], dtype=np.float32)
    tfeat = torch.tensor([1.0 if is_consistent(r[0], r[3], seed) else 0.0 for r in rows], dtype=torch.float32)
    vic = [r[1] for r in rows]
    order = sorted(set(vic))
    te_v = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order)//3)]])
    te = np.array([x in te_v for x in vic]); tr = ~te
    tr_t, te_t = torch.tensor(tr), torch.tensor(te)
    yt = torch.tensor(y)

    # --- GNN ---
    model = MultiplexGNN(len(twins), len(names))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    lossf = nn.BCEWithLogitsLoss()
    for _ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        logit = model(adjs, s_idx, v_idx, tfeat)
        loss = lossf(logit[tr_t], yt[tr_t])
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        sc = torch.sigmoid(model(adjs, s_idx, v_idx, tfeat))[te_t].numpy()
    gnn_auc = roc_auc_score(y[te], sc); gnn_r1 = _recall_at_fpr(y[te], sc)

    # --- baseline LightGBM (reczne cechy) ---
    X = np.array([[1.0 if r[0] in {x for x in _nb(layers[nm], r[1])} else 0.0 for nm in names]
                  + [1.0 if is_consistent(r[0], r[3], seed) else 0.0] for r in rows], dtype=np.float32)
    clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(X[tr], y[tr])
    sb = clf.predict_proba(X[te])[:, 1]
    lgb_auc = roc_auc_score(y[te], sb); lgb_r1 = _recall_at_fpr(y[te], sb)
    return gnn_auc, gnn_r1, lgb_auc, lgb_r1


_NBCACHE = {}
def _nb(edges, v):
    key = id(edges)
    if key not in _NBCACHE:
        nbr = {}
        for e in edges:
            nbr.setdefault(e[0], set()).add(e[1]); nbr.setdefault(e[1], set()).add(e[0])
        _NBCACHE[key] = nbr
    return _NBCACHE[key].get(v, set())


def main():
    acc = {"gnn_auc": [], "gnn_r1": [], "lgb_auc": [], "lgb_r1": []}
    for seed in SEEDS:
        ga, gr, la, lr = run(seed)
        acc["gnn_auc"].append(ga); acc["gnn_r1"].append(gr)
        acc["lgb_auc"].append(la); acc["lgb_r1"].append(lr)
        print(f"  seed={seed}: GNN auc={ga:.3f} r1={gr:.3f} | LGBM auc={la:.3f} r1={lr:.3f}", flush=True)
    agg = {k: float(np.mean(v)) for k, v in acc.items()}
    out = RESULTS / "exp_gnn.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["metric", "mean", "n_seeds"])
        for k, v in acc.items():
            w.writerow([k, round(float(np.mean(v)), 4), len(v)])
    print(f"\n[gnn] GNN  AUC={agg['gnn_auc']:.3f} R@1%={agg['gnn_r1']:.3f}")
    print(f"[gnn] LGBM AUC={agg['lgb_auc']:.3f} R@1%={agg['lgb_r1']:.3f}")
    print(f"[gnn] wrote {out}")


if __name__ == "__main__":
    main()
