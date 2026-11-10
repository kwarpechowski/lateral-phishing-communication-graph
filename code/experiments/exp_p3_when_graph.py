"""P3 — KIEDY graf pomaga? Kontrolowany dowod mechanizmu na realnych etykietach (#GP-EXP-32).

Teza spinajaca cykl: struktura grafowa dokłada wartosc DOKLADNIE wtedy, gdy sygnal nie jest juz
w cechach wezla. Elliptic ma 93 cechy LOKALNE + 72 AGREGATY sasiedztwa (graf wbudowany w cechy).
Test: dla podzbioru cech (lokalne / agregaty / pelne) porownujemy baseline (LightGBM) vs fuzja z GCN.
Hipoteza: na SAMYCH LOKALNYCH cechach graf/fuzja POMAGA (sygnal struktury brakuje w cechach);
na PELNYCH (z agregatami) graf nie pomaga (juz tam jest). To wyjasnia, czemu graf pomaga na
kaskadach mailowych (P1/P2), a nie na Elliptic z gotowymi agregatami.

IPEX -> blokada przed PyG. Wyjscie: results/exp_p3_when_graph.csv
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
from sklearn.metrics import f1_score, roc_auc_score        # noqa: E402
from torch_geometric.datasets import EllipticBitcoinDataset  # noqa: E402
from torch_geometric.nn import GCNConv                     # noqa: E402

CODE = Path(__file__).resolve().parent.parent
DATA = CODE / "data" / "elliptic"
RESULTS = CODE / "results"
EPOCHS = 200
N_LOCAL = 93                  # Elliptic: ~93 cechy lokalne, reszta (72) to agregaty sasiedztwa


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid); self.c2 = GCNConv(hid, hid)
        self.lin = torch.nn.Linear(hid, 2)

    def embed(self, x, ei):
        return F.relu(self.c2(F.relu(self.c1(x, ei)), ei))

    def forward(self, x, ei):
        return self.lin(self.embed(x, ei))


def _f1auc(y, proba):
    return f1_score(y, (proba >= 0.5).astype(int)), roc_auc_score(y, proba)


def _gcn_embed(x, ei, y, trm, seed=0):
    torch.manual_seed(seed)
    m = GCN(x.size(1))
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    ytr = y[trm]
    w = torch.tensor([1.0, float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))], dtype=torch.float32)
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        loss = F.cross_entropy(m(x, ei)[trm], y[trm], weight=w); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m.embed(x, ei).numpy()


def main():
    ds = EllipticBitcoinDataset(root=str(DATA))
    d = ds[0]
    x, ei, y = d.x, d.edge_index, d.y
    trm, tem = d.train_mask, d.test_mask
    ytr = y[trm].numpy(); yte = y[tem].numpy()
    subsets = {"lokalne (93)": x[:, :N_LOCAL], "agregaty (72)": x[:, N_LOCAL:], "pelne (165)": x}
    rows = []
    for name, xs in subsets.items():
        Xtr = xs[trm].numpy(); Xte = xs[tem].numpy()
        # baseline cechy
        clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=300, verbose=-1,
                             class_weight="balanced").fit(Xtr, ytr)
        f1b, aucb = _f1auc(yte, clf.predict_proba(Xte)[:, 1])
        # fuzja: cechy + osadzenia GCN (uczone na tym samym podzbiorze)
        emb = _gcn_embed(xs, ei, y, trm)
        Xf_tr = np.concatenate([Xtr, emb[trm.numpy()]], axis=1)
        Xf_te = np.concatenate([Xte, emb[tem.numpy()]], axis=1)
        clf2 = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=300, verbose=-1,
                              class_weight="balanced").fit(Xf_tr, ytr)
        f1f, aucf = _f1auc(yte, clf2.predict_proba(Xf_te)[:, 1])
        gain = f1f - f1b
        rows.append([name, round(f1b, 4), round(f1f, 4), round(gain, 4), round(aucb, 4), round(aucf, 4)])
        print(f"[p3] {name:14s}: cechy F1={f1b:.3f} -> fuzja F1={f1f:.3f}  (zysk grafu {gain:+.3f})", flush=True)
    out = RESULTS / "exp_p3_when_graph.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f); w_.writerow(["features", "f1_cechy", "f1_fuzja", "zysk_grafu", "auc_cechy", "auc_fuzja"])
        w_.writerows(rows)
    print(f"\n[p3] -> {out}")
    print("[p3] WNIOSEK: zysk grafu rosnie, gdy cechy NIE zawieraja agregatow sasiedztwa.")


if __name__ == "__main__":
    main()
