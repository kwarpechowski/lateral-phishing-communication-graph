"""cascadebench.detect — rodziny detektorow ze wspolnym interfejsem.

Detector.fit_score(scenario, train_mask, test_mask, seed) -> wyniki (score'y) na zdarzeniach testowych.
Rodziny: 1-hop (lokalny), COMPA (wolumen per-konto), kontekst (reczny burst), GNN statyczny, GNN temporalny.
Latwo dodac nowy detektor: podklasa z fit_score. To kosc poligonu (rozszerzalnosc).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier

from . import features as F
from . import models as M
from .attack import Scenario

EPOCHS = 100


class Detector:
    name = "base"

    def fit_score(
        self, sc: Scenario, tr: np.ndarray, te: np.ndarray, seed: int
    ) -> np.ndarray:
        raise NotImplementedError


class _LGBMDetector(Detector):
    def _X(self, sc, seed):
        raise NotImplementedError

    def fit_score(self, sc, tr, te, seed):
        X = self._X(sc, seed)
        y = np.array([e[3] for e in sc.events], dtype=np.float32)
        clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(
            X[tr], y[tr]
        )
        return clf.predict_proba(X[te])[:, 1]


class OneHop(_LGBMDetector):
    name = "1-hop"

    def _X(self, sc, seed):
        return F.onehop(sc.graph, sc.events, seed)


class COMPA(_LGBMDetector):
    name = "COMPA"

    def _X(self, sc, seed):
        return F.compa(sc.events)


class HandContext(_LGBMDetector):
    name = "reczny-kontekst"

    def _X(self, sc, seed):
        return F.context(sc.graph, sc.events, seed)


class Hopper(_LGBMDetector):
    """Detektor w duchu Hoppera (Ho i in. 2021) --- sciezka ruchu: precursor przyczynowy + rozmach
    + rzadki cel + brak benign wyjasnienia. Adaptacja do graf+czas (bez tresci maila)."""

    name = "hopper"

    def _X(self, sc, seed):
        return F.hopper(sc.graph, sc.events, seed)


class _TorchDetector(Detector):
    def _tensors(self, sc, seed, tr=None):
        g = sc.graph
        n = len(g)
        idx = g.index
        X1 = F.onehop(g, sc.events, seed)
        Xn = F.node_features(g, sc.events, train_mask=tr)
        s_idx = torch.tensor([idx[e[0]] for e in sc.events])
        v_idx = torch.tensor([idx[e[1]] for e in sc.events])
        ef = torch.tensor(X1, dtype=torch.float32)
        y = torch.tensor([e[3] for e in sc.events], dtype=torch.float32)
        return g, n, Xn, s_idx, v_idx, ef, y


class StaticGNN(_TorchDetector):
    name = "GCN-statyczny"

    def fit_score(self, sc, tr, te, seed):
        torch.manual_seed(seed)
        g, n, Xn, s_idx, v_idx, ef, y = self._tensors(sc, seed, tr)
        A = M.norm_adj(g, n)
        model = M.StaticGNN(Xn.shape[1], ef.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        lf = nn.BCEWithLogitsLoss()
        trt = torch.tensor(tr)
        for _ in range(EPOCHS):
            model.train()
            opt.zero_grad()
            lf(model(A, Xn, s_idx, v_idx, ef)[trt], y[trt]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(A, Xn, s_idx, v_idx, ef))[
                torch.tensor(te)
            ].numpy()


class TemporalGNN(_TorchDetector):
    name = "temporalny-GNN"

    def _make(self, node_dim, edge_dim):
        return M.TemporalGNN(node_dim, edge_dim)

    def fit_score(self, sc, tr, te, seed):
        torch.manual_seed(seed)
        g, n, Xn, s_idx, v_idx, ef, y = self._tensors(sc, seed, tr)
        pb = M.event_batches(g, sc.events)
        model = self._make(Xn.shape[1], ef.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        lf = nn.BCEWithLogitsLoss()
        trt = torch.tensor(tr)
        for _ in range(EPOCHS):
            model.train()
            opt.zero_grad()
            lf(model(Xn, pb, n, ef, s_idx, v_idx)[trt], y[trt]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(Xn, pb, n, ef, s_idx, v_idx))[
                torch.tensor(te)
            ].numpy()


class TemporalGNNAttn(TemporalGNN):
    name = "temporalny-GNN-uwaga"

    def _make(self, node_dim, edge_dim):
        return M.TGATLite(node_dim, edge_dim)


class AnomalyForest(Detector):
    """Nienadzorowany: IsolationForest na cechach kontekstu, uczony na benign, score = anomalia."""

    name = "anomaly-forest"

    def fit_score(self, sc, tr, te, seed):
        from sklearn.ensemble import IsolationForest

        X = F.context(sc.graph, sc.events, seed)
        y = np.array([e[3] for e in sc.events])
        ben_tr = tr & (y == 0)
        clf = IsolationForest(random_state=42, n_estimators=150, contamination="auto")
        clf.fit(X[ben_tr] if ben_tr.sum() > 10 else X[tr])
        return -clf.decision_function(X[te])  # wyzsze = bardziej anomalne = atak


# standardowy panel poligonu (8 rodzin detektorow, w tym Hopper-style SOTA)
PANEL = [
    OneHop,
    COMPA,
    Hopper,
    AnomalyForest,
    StaticGNN,
    HandContext,
    TemporalGNN,
    TemporalGNNAttn,
]


def get_panel():
    return [d() for d in PANEL]
