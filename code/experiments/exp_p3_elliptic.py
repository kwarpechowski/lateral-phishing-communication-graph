"""P3 — walidacja na REALNYCH etykietach: Elliptic Bitcoin (#GP-EXP-30).

Realny, etykietowany, temporalny graf transakcji (legal/illicit) z PyTorch Geometric.
Zero zbierania danych, zero badan na ludziach. Pierwszy krok P3: zaladuj, sprawdz ksztalt,
ustaw baseline (LightGBM na cechach wezla) na PRAWDZIWYCH etykietach -> punkt odniesienia.

IPEX niezgodny z torch 2.12 -> blokada przed importem PyG.
Wyjscie: stdout (ksztalt + baseline F1/AUC dla klasy illicit).
"""
from __future__ import annotations

import sys
sys.modules['intel_extension_for_pytorch'] = None        # noqa: E402

from pathlib import Path                                   # noqa: E402

import numpy as np                                         # noqa: E402
import torch                                               # noqa: E402
from lightgbm import LGBMClassifier                        # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score  # noqa: E402
from torch_geometric.datasets import EllipticBitcoinDataset  # noqa: E402

CODE = Path(__file__).resolve().parent.parent
DATA = CODE / "data" / "elliptic"


def main():
    print("[p3] ladowanie EllipticBitcoinDataset (pierwszy raz pobiera)...", flush=True)
    ds = EllipticBitcoinDataset(root=str(DATA))
    d = ds[0]
    print(f"[p3] wezly={d.num_nodes}  krawedzie={d.num_edges}  cechy={d.num_node_features}", flush=True)
    y = d.y.numpy()
    # Elliptic: y=0 illicit, y=1 licit, y=2 unknown (PyG mapuje: 0=illicit,1=licit; unknown maska)
    tr_mask = d.train_mask.numpy(); te_mask = d.test_mask.numpy()
    print(f"[p3] train={tr_mask.sum()} test={te_mask.sum()}  "
          f"illicit(train)={(y[tr_mask]==1).sum() if False else (d.y[d.train_mask]==1).sum().item()}", flush=True)
    # rozklad etykiet (na zbiorze etykietowanym)
    lab = d.y[d.train_mask | d.test_mask].numpy()
    uniq, cnt = np.unique(lab, return_counts=True)
    print(f"[p3] rozklad etykiet (labeled): {dict(zip(uniq.tolist(), cnt.tolist()))}  "
          f"(1=illicit, 0=licit w PyG)", flush=True)

    X = d.x.numpy()
    ytr = d.y[d.train_mask].numpy(); yte = d.y[d.test_mask].numpy()
    Xtr = X[tr_mask]; Xte = X[te_mask]
    # baseline: LightGBM na cechach wezla (standardowy punkt odniesienia Elliptic)
    clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=300, verbose=-1,
                         class_weight="balanced").fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]; pred = (proba >= 0.5).astype(int)
    # klasa 1 = illicit (pozytywna)
    print("\n[p3] BASELINE LightGBM na REALNYCH etykietach (klasa illicit=1):")
    print(f"  AUC      = {roc_auc_score(yte, proba):.3f}")
    print(f"  F1       = {f1_score(yte, pred):.3f}")
    print(f"  Precision= {precision_score(yte, pred):.3f}")
    print(f"  Recall   = {recall_score(yte, pred):.3f}")
    print(f"  illicit w tescie: {int((yte == 1).sum())}/{len(yte)}")


if __name__ == "__main__":
    main()
