"""P3 — cechy vs graf vs fuzja na REALNYCH etykietach Elliptic (#GP-EXP-31).

Pytanie: czy sygnal grafowy (propagacja) dokłada wartosc ponad mocne cechy wezla na realnie
etykietowanym, temporalnym grafie transakcji? Porownanie:
  * cechy        : LightGBM na 165 cechach wezla (baseline literaturowy)
  * GCN          : 2-warstwowy GCN (uczona reprezentacja grafowa)
  * fuzja        : LightGBM na [cechy + osadzenia GCN]  (analog multipleksu P1 / kontekstu P2)
Metryka: F1/AUC/precision/recall dla klasy illicit (rzadka). Podzial CZASOWY (z datasetu).

IPEX niezgodny z torch 2.12 -> blokada przed importem PyG.
Wyjscie: results/exp_p3_elliptic.csv
"""
from __future__ import annotations

import sys
sys.modules['intel_extension_for_pytorch'] = None        # noqa: E402

import csv                                                 # noqa: E402
from pathlib import Path                                   # noqa: E402

import numpy as np                                         # noqa: E402
import torch                                               # noqa: E402
import torch.nn.functional as F                            # noqa: E402
from lightgbm import LGBMClassifier                        # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score  # noqa: E402
from torch_geometric.datasets import EllipticBitcoinDataset  # noqa: E402
from torch_geometric.nn import GCNConv                     # noqa: E402

CODE = Path(__file__).resolve().parent.parent
DATA = CODE / "data" / "elliptic"
RESULTS = CODE / "results"
EPOCHS = 200
HID = 64


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid=HID, out=2):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.c2 = GCNConv(hid, hid)
        self.lin = torch.nn.Linear(hid, out)

    def embed(self, x, ei):
        h = F.relu(self.c1(x, ei))
        h = F.relu(self.c2(h, ei))
        return h

    def forward(self, x, ei):
        return self.lin(self.embed(x, ei))


def _metrics(y, proba):
    pred = (proba >= 0.5).astype(int)
    return {"auc": roc_auc_score(y, proba), "f1": f1_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0)}


def main():
    ds = EllipticBitcoinDataset(root=str(DATA))
    d = ds[0]
    x, ei, y = d.x, d.edge_index, d.y
    trm, tem = d.train_mask, d.test_mask
    Xtr = x[trm].numpy(); Xte = x[tem].numpy()
    ytr = y[trm].numpy(); yte = y[tem].numpy()
    rows = []

    # --- cechy (baseline) ---
    clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=300, verbose=-1,
                         class_weight="balanced").fit(Xtr, ytr)
    m = _metrics(yte, clf.predict_proba(Xte)[:, 1])
    rows.append(["cechy", m]); print(f"[p3] cechy: F1={m['f1']:.3f} AUC={m['auc']:.3f}", flush=True)

    # --- GCN ---
    torch.manual_seed(0)
    model = GCN(x.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    w = torch.tensor([1.0, float((ytr == 0).sum()) / max(1, (ytr == 1).sum())],
                     dtype=torch.float32)  # waga klasy illicit
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        out = model(x, ei)
        loss = F.cross_entropy(out[trm], y[trm], weight=w)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logit = model(x, ei)
        proba_gcn = F.softmax(logit, dim=1)[:, 1].numpy()
        emb = model.embed(x, ei).numpy()
    m = _metrics(yte, proba_gcn[tem.numpy()])
    rows.append(["gcn", m]); print(f"[p3] GCN: F1={m['f1']:.3f} AUC={m['auc']:.3f}", flush=True)

    # --- fuzja: cechy + osadzenia GCN ---
    Xf_tr = np.concatenate([Xtr, emb[trm.numpy()]], axis=1)
    Xf_te = np.concatenate([Xte, emb[tem.numpy()]], axis=1)
    clf2 = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=300, verbose=-1,
                          class_weight="balanced").fit(Xf_tr, ytr)
    m = _metrics(yte, clf2.predict_proba(Xf_te)[:, 1])
    rows.append(["fuzja", m]); print(f"[p3] fuzja (cechy+GCN): F1={m['f1']:.3f} AUC={m['auc']:.3f}", flush=True)

    out = RESULTS / "exp_p3_elliptic.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f); w_.writerow(["model", "auc", "f1", "precision", "recall"])
        for name, mm in rows:
            w_.writerow([name, round(mm["auc"], 4), round(mm["f1"], 4),
                         round(mm["precision"], 4), round(mm["recall"], 4)])
    print(f"\n[p3] -> {out}")
    print("[p3] podsumowanie (illicit):")
    for name, mm in rows:
        print(f"  {name:7s} AUC={mm['auc']:.3f} F1={mm['f1']:.3f} "
              f"P={mm['precision']:.3f} R={mm['recall']:.3f}")


if __name__ == "__main__":
    main()
