"""Uczony multipleksowy GNN z samonadzorowanym celem spojnosci + split indukcyjny.

Program A rozprawy (#GP-EXP-14 + #GP-EXP-15). Dwa pytania:

  EXP-14 (cel spojnosci): czy strata L = L_cls + lambda * L_spoj bije lambda=0
    i reczne cechy? L_spoj = samonadzorowana rekonstrukcja krawedzi PER WARSTWA
    (UMGAD) + sciaganie osadzen warstw do konsensusu (DMGI). Anomalia zdarzenia =
    duza niespojnosc miedzywarstwowa nadawcy.

  EXP-15 (generalizacja indukcyjna): czy model dziala na NIEWIDZIANE organizacje?
    Transduktywny ID-GNN (nn.Embedding per wezel, baseline EXP-8) nie potrafi --
    test-wezly maja nieuczone osadzenia. Featuralny GNN (cechy strukturalne wezla:
    stopien per-warstwa) jest architektonicznie indukcyjny. Split: organizacje
    rozlaczne train/test (19 firm -> ~13 train / ~6 test). Raportujemy luke transferu.

Modele: ID-GNN (transd.) | Feat-GNN(lambda) (indukcyjny) | LightGBM (reczne cechy).
Tryby splitu: 'victim' (po ofiarach, jak EXP-8) | 'org' (po organizacjach, indukcyjny).

Wyjscie: results/exp_gnn_consistency.csv
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
EPOCHS = 200
LAMBDAS = [0.0, 0.1, 0.5, 1.0]      # ablacja celu spojnosci
# Diagnostyk: 'notfeat' usuwa bit spojnosci czasowej z WEJSCIA klasyfikatora -> model
# musi wyciagnac czas z warstwy temporal_coactivity przez message-passing (osadzenia maja
# wtedy znaczenie; pytanie indukcyjne staje sie sensowne).
NO_TFEAT = "notfeat" in sys.argv
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


def _org_map():
    """twin_id -> organizacja (do splitu indukcyjnego)."""
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        return {r["twin_id"]: r["org"] for r in csv.DictReader(f)}


def _self_rows():
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _norm_adj(edges, idx, n):
    A = torch.eye(n)
    for a, b in edges:
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            A[i, j] = 1.0; A[j, i] = 1.0
    d = A.sum(1, keepdim=True).clamp(min=1.0)
    return A / d


def _norm_adj_masked(edges, idx, n, drop):
    """Znormalizowana adjacencja z USUNIETYMI wezlami `drop` (zbior twin_id) -- graf treningowy
    trybu strict: wezly testowych firm znikaja z message-passing."""
    A = torch.eye(n)
    for a, b in edges:
        if a in idx and b in idx and a not in drop and b not in drop:
            i, j = idx[a], idx[b]
            A[i, j] = 1.0; A[j, i] = 1.0
    for t in drop:
        if t in idx:
            A[idx[t], :] = 0.0
    d = A.sum(1, keepdim=True).clamp(min=1.0)
    return A / d


def _raw_adj(edges, idx, n, drop=frozenset()):
    """Niesymetryzowana macierz 0/1 (bez self-loop) do rekonstrukcji; opcjonalnie bez `drop`."""
    A = torch.zeros(n, n)
    for a, b in edges:
        if a in idx and b in idx and a not in drop and b not in drop:
            i, j = idx[a], idx[b]
            A[i, j] = 1.0; A[j, i] = 1.0
    return A


def _node_features(layers, names, idx, n):
    """Cechy strukturalne wezla (INDUKCYJNE): stopien per-warstwa, log1p, standaryzacja.
    Istnieja tez dla wezlow niewidzianych w treningu -> umozliwiaja transfer cross-org.
    """
    X = torch.zeros(n, len(names))
    for li, nm in enumerate(names):
        nbr = {}
        for a, b in layers[nm]:
            nbr.setdefault(a, set()).add(b); nbr.setdefault(b, set()).add(a)
        for t, i in idx.items():
            X[i, li] = len(nbr.get(t, ()))
    X = torch.log1p(X)
    X = (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp(min=1e-6)
    return X


def _events_and_graph(seed):
    """Generacja zdarzen + warstwy multipleksu (identyczny model danych jak EXP-8)."""
    base = build_layers()
    ov, _ = build_overlay(seed=seed, p_cross=0.5, fabrication_rate=0.0)
    twins_all = sorted({r["twin_id"] for r in _self_rows()})
    layers = {"contact": base["contact"], **ov,
              "temporal_coactivity": coactivity_edges(twins_all, seed)}
    names = list(layers)
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


# ----------------------------- modele -----------------------------------------

class IDGNN(nn.Module):
    """Baseline transduktywny: uczone osadzenia per wezel (jak EXP-8)."""
    def __init__(self, n_nodes, n_layers, dim=DIM):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, dim)
        self.w1 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.w2 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.clf = nn.Sequential(nn.Linear(dim * n_layers * 3 + 1, 64), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(64, 1))

    def reps(self, adjs):
        X = self.emb.weight
        return [A @ w2(torch.relu(A @ w1(X))) for A, w1, w2 in zip(adjs, self.w1, self.w2)]

    def forward(self, adjs, s_idx, v_idx, tfeat):
        H = torch.cat(self.reps(adjs), dim=1)
        hs, hv = H[s_idx], H[v_idx]
        e = torch.cat([hs, hv, hs * hv, tfeat.unsqueeze(1)], dim=1)
        return self.clf(e).squeeze(1)


class FeatGNN(nn.Module):
    """Indukcyjny: cechy strukturalne wezla -> per-warstwa propagacja -> konsensus.
    Wspiera samonadzorowany cel spojnosci (rekonstrukcja per-warstwa + konsensus).
    """
    def __init__(self, in_dim, n_layers, dim=DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.w1 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.w2 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        # klasyfikator: konsensus(s,v) + iloczyn + niespojnosc nadawcy + cecha czasu
        self.clf = nn.Sequential(nn.Linear(dim * 3 + 2, 64), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(64, 1))

    def layer_reps(self, X, adjs):
        h0 = torch.relu(self.proj(X))
        return [A @ w2(torch.relu(A @ w1(h0))) for A, w1, w2 in zip(adjs, self.w1, self.w2)]

    def forward(self, X, adjs, s_idx, v_idx, tfeat):
        hl = self.layer_reps(X, adjs)                       # lista [n,dim] per warstwa
        Hs = torch.stack(hl, dim=0)                         # [L, n, dim]
        z = Hs.mean(0)                                      # konsensus [n, dim]
        disagree = ((Hs - z) ** 2).mean(0).mean(1)          # niespojnosc per wezel [n]
        zs, zv = z[s_idx], z[v_idx]
        e = torch.cat([zs, zv, zs * zv,
                       disagree[s_idx].unsqueeze(1), tfeat.unsqueeze(1)], dim=1)
        return self.clf(e).squeeze(1), hl, z

    def consistency_loss(self, hl, z, raw_adjs, train_nodes, gen):
        """Samonadzor: (a) rekonstrukcja krawedzi per warstwa (UMGAD), (b) konsensus (DMGI)."""
        recon = torch.tensor(0.0)
        tn = train_nodes
        for h, A in zip(hl, raw_adjs):
            # probkowanie: wszystkie krawedzie pos w obrebie train + tyle samo neg
            sub = A[tn][:, tn]
            logits = h[tn] @ h[tn].t()
            pos = sub > 0
            npos = int(pos.sum().item())
            if npos == 0:
                continue
            neg_mask = sub == 0
            negflat = neg_mask.flatten().nonzero(as_tuple=False).squeeze(1)
            pick = negflat[torch.randint(len(negflat), (npos,), generator=gen)]
            negsel = torch.zeros_like(sub.flatten()); negsel[pick] = 1.0
            negsel = negsel.reshape(sub.shape) > 0
            lp = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[pos], torch.ones(npos))
            ln = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[negsel], torch.zeros(int(negsel.sum().item())))
            recon = recon + 0.5 * (lp + ln)
        recon = recon / max(1, len(hl))
        Hs = torch.stack(hl, dim=0)
        consensus = ((Hs[:, tn] - z[tn]) ** 2).mean()
        return recon + consensus


# ----------------------------- ewaluacja --------------------------------------

def _split(rows, twins, mode, seed):
    """Maska train/test zdarzen + zbior wezlow testowych (dla strict).
    mode='victim' (po ofiarach) | 'org' (po org, label-only) | 'org_strict'
    (po org, wezly test USUNIETE z grafu treningowego -> prawdziwie indukcyjny).
    Zwraca (tr_mask, te_mask, test_nodes:set).
    """
    rng = np.random.default_rng(seed)
    if mode == "victim":
        vic = sorted({r[1] for r in rows})
        te_keys = set(np.array(vic)[rng.permutation(len(vic))[:max(1, len(vic) // 3)]])
        te = np.array([r[1] in te_keys for r in rows])
        return ~te, te, set()
    omap = _org_map()
    orgs = sorted({omap[t] for t in twins})
    te_orgs = set(np.array(orgs)[rng.permutation(len(orgs))[:max(1, len(orgs) // 3)]])
    test_nodes = {t for t in twins if omap[t] in te_orgs}
    if mode == "org":
        te = np.array([omap[r[1]] in te_orgs for r in rows])
        return ~te, te, set()
    # org_strict: test = ofiara w test-org; trening = zdarzenia BEZ zadnego wezla testowego
    te = np.array([r[1] in test_nodes for r in rows])
    tr = np.array([(r[0] not in test_nodes) and (r[1] not in test_nodes) for r in rows])
    return tr, te, test_nodes


def _hand_features(rows, layers, names, seed):
    X = np.array([[1.0 if r[0] in _nb(layers[nm], r[1]) else 0.0 for nm in names]
                  + [1.0 if is_consistent(r[0], r[3], seed) else 0.0] for r in rows],
                 dtype=np.float32)
    return X


_NBCACHE = {}
def _nb(edges, v):
    key = id(edges)
    if key not in _NBCACHE:
        nbr = {}
        for e in edges:
            nbr.setdefault(e[0], set()).add(e[1]); nbr.setdefault(e[1], set()).add(e[0])
        _NBCACHE[key] = nbr
    return _NBCACHE[key].get(v, set())


def _train_idgnn(adjs_tr, adjs_ev, s_idx, v_idx, tfeat, yt, tr_t, n, nL):
    model = IDGNN(n, nL)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        loss = lossf(model(adjs_tr, s_idx, v_idx, tfeat)[tr_t], yt[tr_t])
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(adjs_ev, s_idx, v_idx, tfeat)).numpy()


def _train_featgnn(X, adjs_tr, adjs_ev, raw_adjs, s_idx, v_idx, tfeat, yt, tr_t, lam, seed, nL):
    model = FeatGNN(X.shape[1], nL)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    lossf = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    train_nodes = torch.unique(torch.cat([s_idx[tr_t], v_idx[tr_t]]))
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        logit, hl, z = model(X, adjs_tr, s_idx, v_idx, tfeat)
        loss = lossf(logit[tr_t], yt[tr_t])
        if lam > 0:
            loss = loss + lam * model.consistency_loss(hl, z, raw_adjs, train_nodes, gen)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X, adjs_ev, s_idx, v_idx, tfeat)[0]).numpy()


def run_seed(seed, mode):
    twins, layers, names, rows = _events_and_graph(seed)
    idx = {t: i for i, t in enumerate(twins)}
    n, nL = len(twins), len(names)
    tr, te, test_nodes = _split(rows, twins, mode, seed)

    adjs_ev = [_norm_adj(layers[nm], idx, n) for nm in names]            # pelny graf (ewaluacja)
    if test_nodes:  # strict: graf treningowy bez wezlow testowych firm
        adjs_tr = [_norm_adj_masked(layers[nm], idx, n, test_nodes) for nm in names]
        raw_adjs = [_raw_adj(layers[nm], idx, n, test_nodes) for nm in names]
    else:
        adjs_tr = adjs_ev
        raw_adjs = [_raw_adj(layers[nm], idx, n) for nm in names]
    X = _node_features(layers, names, idx, n)

    s_idx = torch.tensor([idx[r[0]] for r in rows])
    v_idx = torch.tensor([idx[r[1]] for r in rows])
    y = np.array([r[2] for r in rows], dtype=np.float32)
    if NO_TFEAT:   # zero -> oracle czasu wylaczony; sygnal musi przejsc przez warstwe/osadzenia
        tfeat = torch.zeros(len(rows), dtype=torch.float32)
    else:
        tfeat = torch.tensor([1.0 if is_consistent(r[0], r[3], seed) else 0.0 for r in rows],
                             dtype=torch.float32)
    yt = torch.tensor(y)
    tr_t, te_t = torch.tensor(tr), torch.tensor(te)

    out = {}
    # ID-GNN (transduktywny): w strict test-wezly maja NIEUCZONE osadzenia -> spodziewany kolaps
    s = _train_idgnn(adjs_tr, adjs_ev, s_idx, v_idx, tfeat, yt, tr_t, n, nL)
    out["idgnn"] = (roc_auc_score(y[te], s[te]), _recall_at_fpr(y[te], s[te], 0.01),
                    _recall_at_fpr(y[te], s[te], 0.001))
    # Feat-GNN dla kazdej lambdy (indukcyjny: cechy strukturalne dzialaja dla niewidzianych)
    for lam in LAMBDAS:
        s = _train_featgnn(X, adjs_tr, adjs_ev, raw_adjs, s_idx, v_idx, tfeat, yt, tr_t,
                           lam, seed, nL)
        out[f"featgnn_l{lam}"] = (roc_auc_score(y[te], s[te]),
                                  _recall_at_fpr(y[te], s[te], 0.01),
                                  _recall_at_fpr(y[te], s[te], 0.001))
    # LightGBM reczne cechy (indukcyjny w przestrzeni cech)
    Xh = _hand_features(rows, layers, names, seed)
    if NO_TFEAT:   # usun ostatnia kolumne (cecha spojnosci czasowej) dla uczciwego porownania
        Xh = Xh[:, :-1]
    clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(Xh[tr], y[tr])
    sb = clf.predict_proba(Xh[te])[:, 1]
    out["lgbm"] = (roc_auc_score(y[te], sb), _recall_at_fpr(y[te], sb, 0.01),
                   _recall_at_fpr(y[te], sb, 0.001))
    return out


def main():
    modes = ["victim", "org_strict"] if NO_TFEAT else ["victim", "org", "org_strict"]
    agg = {m: {} for m in modes}
    for mode in modes:
        print(f"\n=== tryb splitu: {mode} ===", flush=True)
        for seed in SEEDS:
            res = run_seed(seed, mode)
            for k, (a, r1, r01) in res.items():
                agg[mode].setdefault(k, {"auc": [], "r1": [], "r01": []})
                agg[mode][k]["auc"].append(a)
                agg[mode][k]["r1"].append(r1)
                agg[mode][k]["r01"].append(r01)
            line = " | ".join(f"{k} auc={v[0]:.3f}" for k, v in res.items())
            print(f"  seed={seed}: {line}", flush=True)

    out = RESULTS / ("exp_gnn_consistency_notfeat.csv" if NO_TFEAT
                     else "exp_gnn_consistency.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "model", "auc", "recall_fpr1", "recall_fpr01", "n_seeds"])
        for mode in modes:
            for k, v in agg[mode].items():
                w.writerow([mode, k, round(float(np.mean(v["auc"])), 4),
                            round(float(np.mean(v["r1"])), 4),
                            round(float(np.mean(v["r01"])), 4), len(v["auc"])])
    print(f"\n[gnn-consistency] zapisano {out}")
    # skrot: luka transferu victim -> org_strict (prawdziwie indukcyjny) dla 3 modeli
    print("  --- luka transferu victim -> org_strict (indukcyjny, wezly test usuniete z grafu) ---")
    for k in ["idgnn", "featgnn_l0.0", "featgnn_l0.5", "lgbm"]:
        if k in agg["victim"] and k in agg["org_strict"]:
            gv = np.mean(agg["victim"][k]["auc"]); gs = np.mean(agg["org_strict"][k]["auc"])
            print(f"  {k}: AUC victim={gv:.3f} -> org_strict={gs:.3f} (luka {gv-gs:+.3f})")


if __name__ == "__main__":
    main()
