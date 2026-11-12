"""Biblioteczna TGN (PyTorch Geometric) jako mocny baseline vs nasza TGN-lite + reczny (#GP-EXP-25).

Domyka ostatni punkt recenzenta: porownanie z PELNA, biblioteczna implementacja TGN [Rossi i in. 2020]
(torch_geometric.nn.models.tgn: TGNMemory + GraphAttentionEmbedding/TransformerConv + LastNeighborLoader).
Adaptacja przykladu PyG (link prediction) do KLASYFIKACJI krawedzi (benign vs zlosliwa kaskada).

Uwaga srodowiskowa: IPEX (intel_extension_for_pytorch) jest niezgodny z torch 2.12 i twardo ubija
proces przy imporcie PyG -> blokujemy go przez sys.modules PRZED importem torch_geometric.

Wyjscie: results/exp_cascade_tgn.csv
"""

from __future__ import annotations

import sys

sys.modules["intel_extension_for_pytorch"] = None  # noqa: E402  (IPEX niezgodny z torch 2.12)

import csv  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.nn import Linear  # noqa: E402
from torch_geometric.nn import TransformerConv  # noqa: E402
from torch_geometric.nn.models.tgn import (  # noqa: E402
    TGNMemory,
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "experiments"))
RESULTS = CODE / "results"

import exp_cascade as XC  # noqa: E402
from exp_cascade import (
    _simulate,
    _features,
    _index_traffic,  # noqa: E402
    _node_features,
    _recall_at_fpr,
    TemporalGNN,
)

SEEDS = list(range(3))
MEM_DIM = 32
TIME_DIM = 32
EMB_DIM = 32
EPOCHS = 50
BATCH = 200
EPOCHS_LITE = 100


class GraphAttentionEmbedding(torch.nn.Module):
    """Modul osadzenia z przykladu PyG TGN: TransformerConv z kodowaniem czasu na krawedzi."""

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkClf(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l1 = Linear(dim * 2, dim)
        self.l2 = Linear(dim, 1)

    def forward(self, zs, zd):
        return self.l2(torch.relu(self.l1(torch.cat([zs, zd], dim=-1)))).squeeze(-1)


def _events_tensors(seed):
    twins, nbr, events = _simulate(seed)
    idx = {t: i for i, t in enumerate(twins)}
    events.sort(key=lambda e: e[2])  # po czasie (bucket)
    src = torch.tensor([idx[e[0]] for e in events])
    dst = torch.tensor([idx[e[1]] for e in events])
    t = torch.tensor(
        [int(e[2]) for e in events], dtype=torch.long
    )  # TGN: timestamp = Long
    y = torch.tensor([float(e[3]) for e in events])
    recv_b, sent_b = _index_traffic(events)
    X1, Xc, _y = _features(events, nbr, recv_b, sent_b, seed)
    msg = torch.tensor(X1, dtype=torch.float32)  # raw message = cechy 1-hop
    vics = sorted({e[1] for e in events})
    rng = np.random.default_rng(seed)
    te_v = set(np.array(vics)[rng.permutation(len(vics))[: max(1, len(vics) // 3)]])
    te = torch.tensor([e[1] in te_v for e in events])
    return twins, nbr, idx, events, src, dst, t, y, msg, te, X1, Xc, recv_b, sent_b


def _run_pyg_tgn(n_nodes, src, dst, t, y, msg, te, seed):
    torch.manual_seed(seed)
    raw_dim = msg.size(-1)
    memory = TGNMemory(
        n_nodes,
        raw_dim,
        MEM_DIM,
        TIME_DIM,
        message_module=IdentityMessage(raw_dim, MEM_DIM, TIME_DIM),
        aggregator_module=LastAggregator(),
    )
    gnn = GraphAttentionEmbedding(MEM_DIM, EMB_DIM, raw_dim, memory.time_enc)
    clf = LinkClf(EMB_DIM)
    opt = torch.optim.Adam(
        list(memory.parameters()) + list(gnn.parameters()) + list(clf.parameters()),
        lr=0.001,
    )
    lossf = torch.nn.BCEWithLogitsLoss()
    nl = LastNeighborLoader(n_nodes, size=10)
    assoc = torch.empty(n_nodes, dtype=torch.long)
    tr = ~te
    E = src.size(0)

    def pass_(train):
        memory.train(train)
        gnn.train(train)
        clf.train(train)
        memory.reset_state()
        nl.reset_state()
        scores = torch.zeros(E)
        for s in range(0, E, BATCH):
            e = slice(s, min(s + BATCH, E))
            sb, db, tb, yb, mb = src[e], dst[e], t[e], y[e], msg[e]
            n_id = torch.cat([sb, db]).unique()
            n_id, edge_index, e_id = nl(n_id)
            assoc[n_id] = torch.arange(n_id.size(0))
            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, t[e_id], msg[e_id])
            out = clf(z[assoc[sb]], z[assoc[db]])
            if train:
                m = tr[e]
                if m.any():
                    opt.zero_grad()
                    loss = lossf(out[m], yb[m])
                    memory.update_state(sb, db, tb, mb)
                    nl.insert(sb, db)
                    loss.backward()
                    opt.step()
                    memory.detach()
                else:
                    memory.update_state(sb, db, tb, mb)
                    nl.insert(sb, db)
            else:
                scores[e] = torch.sigmoid(out.detach())
                memory.update_state(sb, db, tb, mb)
                nl.insert(sb, db)
        return scores

    for _ in range(EPOCHS):
        pass_(True)
    with torch.no_grad():
        sc = pass_(False)
    yt, tem = y.numpy(), te.numpy()
    return (
        roc_auc_score(yt[tem], sc.numpy()[tem]),
        _recall_at_fpr(yt[tem], sc.numpy()[tem]),
    )


def run_seed(seed):
    (twins, nbr, idx, events, src, dst, t, y, msg, te, X1, Xc, recv_b, sent_b) = (
        _events_tensors(seed)
    )
    n = len(twins)
    yt = y.numpy()
    tem = te.numpy()
    trm = ~tem
    out = {}
    # reczny kontekst
    c = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(
        Xc[trm], yt[trm]
    )
    sc = c.predict_proba(Xc[tem])[:, 1]
    out["tab_ctx"] = (roc_auc_score(yt[tem], sc), _recall_at_fpr(yt[tem], sc))
    # nasza TGN-lite
    torch.manual_seed(seed)
    Xn = _node_features(events, nbr, idx, n, recv_b, sent_b, tr=trm)
    ef = torch.tensor(X1, dtype=torch.float32)
    m = TemporalGNN(Xn.shape[1], X1.shape[1])
    pb = XC._event_batches_local(events, idx)  # przyczynowosc zdarzeniowa (fix A1)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    lf = torch.nn.BCEWithLogitsLoss()
    trt = torch.tensor(trm)
    for _ in range(EPOCHS_LITE):
        m.train()
        opt.zero_grad()
        loss = lf(m(Xn, pb, n, ef, src, dst)[trt], y[trt])
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        sl = torch.sigmoid(m(Xn, pb, n, ef, src, dst)).numpy()
    out["tgn_lite"] = (
        roc_auc_score(yt[tem], sl[tem]),
        _recall_at_fpr(yt[tem], sl[tem]),
    )
    # biblioteczna PyG TGN
    out["tgn_pyg"] = _run_pyg_tgn(n, src, dst, t, y, msg, te, seed)
    return out


def main():
    agg = {}
    for seed in SEEDS:
        res = run_seed(seed)
        for k, (a, r) in res.items():
            agg.setdefault(k, {"a": [], "r": []})
            agg[k]["a"].append(a)
            agg[k]["r"].append(r)
        print(
            "  seed=%d: " % seed
            + " | ".join(f"{k} auc={v[0]:.3f} r1={v[1]:.3f}" for k, v in res.items()),
            flush=True,
        )
    out = RESULTS / "exp_cascade_tgn.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "auc", "recall_fpr1", "n_seeds"])
        for k, v in agg.items():
            w.writerow(
                [
                    k,
                    round(float(np.mean(v["a"])), 4),
                    round(float(np.mean(v["r"])), 4),
                    len(v["a"]),
                ]
            )
    print(f"\n[tgn] -> {out}")
    for k in ["tab_ctx", "tgn_lite", "tgn_pyg"]:
        if k in agg:
            print(
                f"  {k}: AUC={np.mean(agg[k]['a']):.3f}  R@FPR1%={np.mean(agg[k]['r']):.3f}"
            )


if __name__ == "__main__":
    main()
