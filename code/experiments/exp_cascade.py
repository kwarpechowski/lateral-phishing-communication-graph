"""Premisa P2: kaskada lateral-movement = sygnal WIELOHOPOWY, ktorego 1-hop nie widzi (#GP-EXP-17).

Diagnostyk EXP-14/15 pokazal: na punktowym modelu zdarzen sygnal jest tabelaryczny (1-hop
membership + bit czasu) -> GNN nie ma czego sie uczyc. Tu modelujemy atak WIERNIEJ: lateral
phishing jako PROPAGACJE na grafie kontaktow (przejety wezel -> jego kontakty -> ich kontakty,
w czasie). Pojedyncze zdarzenie ataku (A->B) wyglada LOKALNIE jak zwykla poczta od znanego
kontaktu (A JEST kontaktem B, moze byc w rytmie). Jedyny sygnal to KONTEKST KASKADY:
synchroniczny burst u sasiadow B + swiezosc "dotkniecia" nadawcy A. To z natury wielohopowe
i czasowe -> model 1-hop powinien byc ~losowy, a kontekst-sasiedztwa / GNN powinien wygrac.

Cel: pokazac, ze premisa P2 trzyma (istnieje sygnal, ktorego cechy 1-hop nie lapia).
Detektory: (1) 1-hop tabelaryczny, (2) kontekst-sasiedztwa tabelaryczny, (3) featuralny GNN
po grafie zdarzen czasowych. Metryki: AUC, Recall@FPR1% dla wykrycia zdarzen kaskady.

WAZNE (brak wycieku etykiety): cechy kontekstu liczone z OBSERWOWALNEGO strumienia wszystkich
wiadomosci (benign+atak, BEZ etykiet) -- burst/recency to wzorzec ruchu, nie ukryty stan infekcji.

Wyjscie: results/exp_cascade.csv
"""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from graph.build_layers import build_layers  # noqa: E402
from graph.build_temporal_overlay import in_bucket, is_consistent  # noqa: E402

SEEDS = list(range(5))
T_BUCKETS = 28  # kroki czasowe (jak rytm w P1)
N_CASCADES = 18  # liczba kaskad (patient-zero) na przebieg
MAX_HOPS = 4  # glebokosc propagacji
P_INFECT = 0.6  # prawdop. ze odbiorca staje sie kolejnym nosicielem
BENIGN_PER_BUCKET = 60  # punktowy ruch legalny na bucket
WINDOW = 2  # okno czasowe dla cech burst/recency
DIM = 24
EPOCHS = 200
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


SPLIT = "org" if "org" in sys.argv else "victim"
# Skala: 'scaleN=<int>' -> graf budowany strukturalnie (org_graph.build_structural), bez LLM.
SCALE_N = next(
    (int(a.split("=")[1]) for a in sys.argv if a.startswith("scaleN=")), None
)

_ORG_CACHE = {}


def _org_map():
    if SCALE_N:
        return _ORG_CACHE  # wypelniany w _contact_graph
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        return {r["twin_id"]: r["org"] for r in csv.DictReader(f)}


def _contact_graph():
    nbr = {}
    if SCALE_N:
        from data.org_graph import build_structural, contacts

        nodes = build_structural(SCALE_N)
        _ORG_CACHE.clear()
        for tid, node in nodes.items():
            _ORG_CACHE[tid] = node["org"]
            for c in contacts(node):
                nbr.setdefault(tid, set()).add(c)
                nbr.setdefault(c, set()).add(tid)
    else:
        base = build_layers()
        for a, b in base["contact"]:
            nbr.setdefault(a, set()).add(b)
            nbr.setdefault(b, set()).add(a)
    twins = sorted(nbr)
    return twins, nbr


def _simulate(seed):
    """Zwraca liste zdarzen (sender, recipient, bucket, label) + graf kontaktow.
    Atak = propagacja kaskady po kontaktach w czasie; benign = punktowy ruch znanych kontaktow."""
    twins, nbr = _contact_graph()
    rng = np.random.default_rng(seed)
    events = []
    # skalowanie ruchu proporcjonalne do N (zachowuje proporcje atak/benign)
    n_casc = N_CASCADES if SCALE_N is None else max(N_CASCADES, len(twins) // 8)
    ben_pb = (
        BENIGN_PER_BUCKET
        if SCALE_N is None
        else max(BENIGN_PER_BUCKET, int(len(twins) * 0.4))
    )

    # --- kaskady (label=1) ---
    seeds0 = [twins[i] for i in rng.permutation(len(twins))[:n_casc]]
    for ci, s0 in enumerate(seeds0):
        t0 = int(rng.integers(0, T_BUCKETS - MAX_HOPS))
        infected = {s0}
        frontier = [s0]
        for hop in range(MAX_HOPS):
            nxt = []
            bucket = t0 + hop
            for u in frontier:
                targets = sorted(nbr.get(u, ()))
                rng.shuffle(targets)
                for v in targets:
                    if v in infected:
                        continue
                    events.append((u, v, bucket, 1))  # malicious send u->v
                    if rng.random() < P_INFECT:
                        infected.add(v)
                        nxt.append(v)
            frontier = nxt
            if not frontier:
                break

    # --- benign punktowy (label=0): znany kontakt, w rytmie ---
    for bucket in range(T_BUCKETS):
        for _ in range(ben_pb):
            v = twins[int(rng.integers(len(twins)))]
            cs = sorted(nbr.get(v, ()))
            if not cs:
                continue
            s = cs[int(rng.integers(len(cs)))]
            events.append((s, v, bucket, 0))
    rng.shuffle(events)
    return twins, nbr, events


def _index_traffic(events):
    """Indeksy OBSERWOWALNE (bez etykiet): kto byl odbiorca / nadawca w danym buckecie."""
    recv_by_bucket = {}  # bucket -> set odbiorcow
    sent_by_bucket = {}  # bucket -> set nadawcow
    for s, v, b, _y in events:
        recv_by_bucket.setdefault(b, set()).add(v)
        sent_by_bucket.setdefault(b, set()).add(s)
    return recv_by_bucket, sent_by_bucket


def _features(events, nbr, recv_by_bucket, sent_by_bucket, seed):
    """Zwraca X_1hop, X_ctx (1hop + kontekst kaskady), y.

    PRZYCZYNOWOSC ZDARZENIOWA (fix po recenzji wewnetrznej): zdarzenie e widzi wylacznie
    zdarzenia o scisle wczesniejszym znaczniku czasu t(e) = (kubelek, pozycja na liscie
    strumienia) — wczesniejsze kubełki okna [b-W, b-1] ORAZ wczesniejsze zdarzenia TEGO
    samego kubełka (regula remisow: porzadek tasowanej listy, losowy ale ustalony).
    Poprzedni podzial po kubełkach (<= b-1) ukrywal fale budujaca sie w biezacym kubełku
    i faworyzowal liczniki wolumetryczne pod blastem (zrodel asymetrii na SNAP)."""
    n = len(events)
    X1 = [None] * n
    Xc = [None] * n
    y = [None] * n
    recv_b: dict = defaultdict(
        set
    )  # kubelek -> odbiorcy (prefix, wypelniany przy przejsciu)
    sent_b: dict = defaultdict(set)  # kubelek -> nadawcy (prefix)
    order = sorted(range(n), key=lambda i: (events[i][2], i))
    for i in order:
        s, v, b, lab = events[i]
        cs = nbr.get(v, set())
        deg_v, deg_s = len(cs), len(nbr.get(s, set()))
        # --- cechy 1-hop (lokalnie atak wyglada normalnie) ---
        f_s_in_contacts = 1.0 if s in cs else 0.0
        f_rhythm = 1.0 if is_consistent(s, b, seed) else 0.0
        f1 = [f_s_in_contacts, f_rhythm, float(deg_v), float(deg_s)]
        # --- kontekst kaskady (OBSERWOWALNY: burst sasiadow + swiezosc nadawcy) ---
        # okno PRZESZLE zdarzeniowe: kompletne kubełki [b-W, b-1] + prefix biezacego b
        recent_recv = set(recv_b.get(b, ()))
        recent_sent = set(sent_b.get(b, ()))
        for bb in range(max(0, b - WINDOW), b):
            recent_recv |= recv_b.get(bb, set())
            recent_sent |= sent_b.get(bb, set())
        burst = len((cs - {s}) & recent_recv) / max(1, deg_v)
        sender_recency = 1.0 if s in recent_recv else 0.0
        prop = len((cs - {s}) & recent_sent) / max(1, deg_v)
        Xc[i] = f1 + [burst, sender_recency, prop]
        X1[i] = f1
        y[i] = float(lab)
        recv_b[b].add(v)
        sent_b[b].add(s)
    return (np.array(X1, np.float32), np.array(Xc, np.float32), np.array(y, np.float32))


# --- featuralny GNN po grafie kontaktow (uczy sie agregowac kontekst) ---
class CascadeGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, dim=DIM):
        super().__init__()
        self.proj = nn.Linear(node_dim, dim)
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.clf = nn.Sequential(
            nn.Linear(dim * 3 + edge_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 1),
        )

    def forward(self, A, Xn, s_idx, v_idx, edge_feat):
        h = torch.relu(self.proj(Xn))
        h = torch.relu(A @ self.w1(h))
        h = A @ self.w2(h)
        e = torch.cat([h[s_idx], h[v_idx], h[s_idx] * h[v_idx], edge_feat], dim=1)
        return self.clf(e).squeeze(1)


class TemporalGNN(nn.Module):
    """TGN-lite: pamiec wezla aktualizowana bucket-po-buckecie z OBSERWOWALNEGO ruchu.
    Zdarzenie w buckecie b klasyfikowane stanem SPRZED aktualizacji b (brak wycieku):
    wezel swiezo 'dotkniety' w b-1 niesie w pamieci sygnal propagacji -> uczy sie kaskady."""

    def __init__(self, node_dim, edge_dim, dim=DIM):
        super().__init__()
        self.proj = nn.Linear(node_dim, dim)
        self.msg = nn.Linear(dim, dim)
        self.upd = nn.GRUCell(dim, dim)
        self.clf = nn.Sequential(
            nn.Linear(dim * 3 + edge_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 1),
        )

    def forward(self, Xn, batches, n, edge_feat, s_all, v_all):
        M = torch.relu(self.proj(Xn))
        logits = torch.zeros(len(s_all))
        for _b, sb, vb, gidx in batches:
            if len(gidx) == 0:
                continue
            # klasyfikacja stanem sprzed aktualizacji bucketa
            e = torch.cat(
                [
                    M[s_all[gidx]],
                    M[v_all[gidx]],
                    M[s_all[gidx]] * M[v_all[gidx]],
                    edge_feat[gidx],
                ],
                dim=1,
            )
            logits = logits.clone()
            logits[gidx] = self.clf(e).squeeze(1)
            # aktualizacja pamieci odbiorcow z nadawcow (ruch, bez etykiet)
            msg = torch.relu(self.msg(M[sb]))
            agg = torch.zeros(n, msg.shape[1]).index_add_(0, vb, msg)
            cnt = (
                torch.zeros(n)
                .index_add_(0, vb, torch.ones(len(vb)))
                .clamp(min=1)
                .unsqueeze(1)
            )
            agg = agg / cnt
            rec = torch.unique(vb)
            M = M.clone()
            M[rec] = self.upd(agg[rec], M[rec])
        return logits


def _norm_adj(nbr, idx, n):
    A = torch.eye(n)
    for u, nbs in nbr.items():
        for w in nbs:
            if u in idx and w in idx:
                A[idx[u], idx[w]] = 1.0
    d = A.sum(1, keepdim=True).clamp(min=1.0)
    return A / d


def _node_features(events, nbr, idx, n, recv_by_bucket, sent_by_bucket, tr=None):
    """Cechy wezla agregowane ze strumienia: laczny in/out + stopien grafu.

    LEAK-AWARE (fix po recenzji): gdy podano maske tr, stopnie strumieniowe liczone
    wylacznie ze zdarzen TRENINGOWYCH (poprzednio caly strumien — w tym zbior testowy
    i przyszle zdarzenia)."""
    X = torch.zeros(n, 3)
    stream = events if tr is None else [e for e, k in zip(events, tr) if k]
    indeg, outdeg = {}, {}
    for s, v, _b, _y in stream:
        outdeg[s] = outdeg.get(s, 0) + 1
        indeg[v] = indeg.get(v, 0) + 1
    for t, i in idx.items():
        X[i, 0] = len(nbr.get(t, ()))
        X[i, 1] = indeg.get(t, 0)
        X[i, 2] = outdeg.get(t, 0)
    X = torch.log1p(X)
    return (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp(min=1e-6)


def _event_batches_local(events, idx, chunk=64):
    """Mini-batche zdarzen w porzadku (kubelek, pozycja na liscie). Batch nie przekracza
    granicy kubełka; zdarzenia w batchu sa wspolczesne (nie widza sie) — przyczynowosc
    zdarzeniowa zamiast kubełkowej (fix po recenzji wewnetrznej; jak cascadebench.models)."""
    by_b = {}
    for i, e in enumerate(events):
        by_b.setdefault(e[2], []).append(i)
    out = []
    for b in sorted(by_b):
        g = by_b[b]
        for c in range(0, len(g), chunk):
            part = g[c : c + chunk]
            sb = torch.tensor([idx[events[i][0]] for i in part])
            vb = torch.tensor([idx[events[i][1]] for i in part])
            out.append((b, sb, vb, part))
    return out


def run_seed(seed):
    twins, nbr, events = _simulate(seed)
    recv_b, sent_b = _index_traffic(events)
    X1, Xc, y = _features(events, nbr, recv_b, sent_b, seed)
    idx = {t: i for i, t in enumerate(twins)}
    n = len(twins)

    # split: po ofiarach (domyslny) lub po ORGANIZACJACH (indukcyjny: niewidziane firmy)
    rng = np.random.default_rng(seed)
    if SPLIT == "org":
        omap = _org_map()
        orgs = sorted({omap[t] for t in twins if t in omap})
        te_orgs = set(
            np.array(orgs)[rng.permutation(len(orgs))[: max(1, len(orgs) // 3)]]
        )
        te = np.array([omap.get(e[1]) in te_orgs for e in events])
        tr = ~te
    else:
        vics = sorted({e[1] for e in events})
        te_v = set(np.array(vics)[rng.permutation(len(vics))[: max(1, len(vics) // 3)]])
        te = np.array([e[1] in te_v for e in events])
        tr = ~te

    out = {}
    # 1-hop tabelaryczny
    c1 = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(
        X1[tr], y[tr]
    )
    s1 = c1.predict_proba(X1[te])[:, 1]
    out["tab_1hop"] = (roc_auc_score(y[te], s1), _recall_at_fpr(y[te], s1))
    # kontekst-sasiedztwa tabelaryczny
    cc = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(
        Xc[tr], y[tr]
    )
    sc = cc.predict_proba(Xc[te])[:, 1]
    out["tab_ctx"] = (roc_auc_score(y[te], sc), _recall_at_fpr(y[te], sc))
    # A3 (recenzja wewnetrzna r2): detektor wolumetryczny COMPA w tym samym
    # przebiegu (przyczynowosc zdarzeniowa) — dla tabeli hierarchii detektorow
    from exp_cascade_sweep import _compa_features as _cf
    Xco = _cf(events)
    cv = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(
        Xco[tr], y[tr]
    )
    sv = cv.predict_proba(Xco[te])[:, 1]
    out["compa"] = (roc_auc_score(y[te], sv), _recall_at_fpr(y[te], sv))
    Xn = _node_features(events, nbr, idx, n, recv_b, sent_b, tr=tr)
    s_idx = torch.tensor([idx[e[0]] for e in events])
    v_idx = torch.tensor([idx[e[1]] for e in events])
    ef = torch.tensor(
        X1, dtype=torch.float32
    )  # cechy krawedzi = 1-hop (bez kontekstu recznego)
    yt = torch.tensor(y)
    tr_t = torch.tensor(tr)
    te_t = torch.tensor(te)
    lossf = nn.BCEWithLogitsLoss()
    # statyczny GNN tylko przy malej skali (gesta macierz A) -- przy SCALE_N pomijany
    if SCALE_N is None:
        A = _norm_adj(nbr, idx, n)
        model = CascadeGNN(Xn.shape[1], X1.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        for _ in range(EPOCHS):
            model.train()
            opt.zero_grad()
            loss = lossf(model(A, Xn, s_idx, v_idx, ef)[tr_t], yt[tr_t])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            sg = torch.sigmoid(model(A, Xn, s_idx, v_idx, ef))[te_t].numpy()
        out["gnn_static"] = (roc_auc_score(y[te], sg), _recall_at_fpr(y[te], sg))

    # --- Temporalny GNN: pamiec aktualizowana per mini-batch zdarzen w porzadku
    #     czasowym (kubelek, pozycja na liscie) — przyczynowosc zdarzeniowa ---
    tmodel = TemporalGNN(Xn.shape[1], X1.shape[1])
    batches = _event_batches_local(events, idx)
    topt = torch.optim.Adam(tmodel.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(80 if SCALE_N else 120):
        tmodel.train()
        topt.zero_grad()
        logit = tmodel(Xn, batches, n, ef, s_idx, v_idx)
        loss = lossf(logit[tr_t], yt[tr_t])
        loss.backward()
        topt.step()
    tmodel.eval()
    with torch.no_grad():
        st = torch.sigmoid(tmodel(Xn, batches, n, ef, s_idx, v_idx))[te_t].numpy()
    out["gnn_temporal"] = (roc_auc_score(y[te], st), _recall_at_fpr(y[te], st))

    if (
        SCALE_N is not None
    ):  # przy skali: kontrola shuffle pominieta (udowodniona przy N=150)
        n_att = int(y.sum())
        n_tot = len(y)
        return out, n_att, n_tot

    # --- KONTROLA PRZYCZYNOWA: przetasowany czas (te same zdarzenia/cechy, losowa kolejnosc
    # bucketow) niszczy strukture kaskady -> przewaga temporalnego GNN musi zniknac ---
    rngs = np.random.default_rng(1000 + seed)
    shuf_b = rngs.integers(0, T_BUCKETS, size=len(events))
    ev_shuf = [(events[i][0], events[i][1], int(shuf_b[i]), events[i][3])
               for i in range(len(events))]
    pb_shuf = _event_batches_local(ev_shuf, idx)
    tmodel2 = TemporalGNN(Xn.shape[1], X1.shape[1])
    topt2 = torch.optim.Adam(tmodel2.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(120):
        tmodel2.train()
        topt2.zero_grad()
        logit = tmodel2(Xn, pb_shuf, n, ef, s_idx, v_idx)
        loss = lossf(logit[tr_t], yt[tr_t])
        loss.backward()
        topt2.step()
    tmodel2.eval()
    with torch.no_grad():
        ss = torch.sigmoid(tmodel2(Xn, pb_shuf, n, ef, s_idx, v_idx))[te_t].numpy()
    out["gnn_temporal_shuf"] = (roc_auc_score(y[te], ss), _recall_at_fpr(y[te], ss))

    n_att = int(y.sum())
    n_tot = len(y)
    return out, n_att, n_tot


def main():
    agg = {}
    perseed = []
    seeds = SEEDS[:3] if SCALE_N else SEEDS
    for seed in seeds:
        res, na, nt = run_seed(seed)
        for k, (a, r) in res.items():
            agg.setdefault(k, {"auc": [], "r1": []})
            agg[k]["auc"].append(a)
            agg[k]["r1"].append(r)
            perseed.append([seed, k, round(a, 4), round(r, 4)])
        line = " | ".join(f"{k} auc={v[0]:.3f} r1={v[1]:.3f}" for k, v in res.items())
        print(f"  seed={seed} (ataki {na}/{nt}): {line}", flush=True)
    tag = f"_scale{SCALE_N}" if SCALE_N else ""
    tag += "_org" if SPLIT == "org" else ""
    out = RESULTS / f"exp_cascade{tag}.csv" if tag else RESULTS / "exp_cascade.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "auc", "recall_fpr1", "n_seeds"])
        for k, v in agg.items():
            w.writerow(
                [
                    k,
                    round(float(np.mean(v["auc"])), 4),
                    round(float(np.mean(v["r1"])), 4),
                    len(v["auc"]),
                ]
            )
    print(f"\n[cascade] zapisano {out}")
    tag_ps = f"_scale{SCALE_N}" if SCALE_N else ""
    tag_ps += "_org" if SPLIT == "org" else ""
    out_ps = RESULTS / f"exp_cascade{tag_ps}_perseed.csv"
    with out_ps.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "model", "auc", "recall_fpr1"])
        w.writerows(perseed)
    print(f"[cascade-perseed] -> {out_ps}")
    for k in ["tab_1hop", "tab_ctx", "gnn_static", "gnn_temporal", "gnn_temporal_shuf"]:
        if k in agg:
            print(
                f"  {k}: AUC={np.mean(agg[k]['auc']):.3f}  Recall@FPR1%={np.mean(agg[k]['r1']):.3f}"
            )


if __name__ == "__main__":
    main()
