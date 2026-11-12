# -*- coding: utf-8 -*-
"""A2: KONTROLA SYGNATURY INIEKCJI na realnych grafach (Enron, SNAP).

Recenzja wewnetrzna: na Enronie i SNAP reżim stealth wydaje się LATWIEJSZY niz blast
(kontekst ręczny rośnie 0.77 -> 0.88). Podejrzenie: wstrzykniete zdarzenia ataku
odrozniaja sie CZASOWO od tla (deterministyczne odstepy miedzy wysylkami), wiec
detektory czytaja sygnature INIEKCJI (artefakt generatora), nie fale odbiorcza.

Trzy narzedzia (wszystkie zapisuja do results/exp_injection_probe.csv):
  1. Rozklady odstepow miedzy wysylkami ataku vs tla (per nadawca i globalnie)
     + dwuprobowy test KS.
  2. Czysto czasowa SONDA: LGBM na cechach wyacznie czasowych nadawcy (odstepy,
     regularnosc, aktywnosc) — wysoki AUC sondy = sygnatura iniekcji.
  3. Wariant 'stealth-empirical': czasy wysylek ataku losowane z EMPIRYCZNEGO
     rozkladu odstepow tla danego grafu; przeliczone Enron i SNAP (COMPA, kontekst
     ręczny, TGN). Jesli po usunieciu sygnatury wyniki sie utrzymaja, fala odbiorcza
     jest realnym sygnalem; jesli znikna — wczesniejszy wynik byl artefaktem.

Uruchomienie: python experiments/exp_injection_probe.py   (z katalogu code/)
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "experiments"))
RESULTS = CODE / "results"

sys.argv = [sys.argv[0]]  # wyczysc argv przed importami (SPLIT/SCALE_N w exp_cascade)

from exp_cascade import (
    TemporalGNN,
    _index_traffic,
    _node_features,  # noqa: E402
    _features,
    _recall_at_fpr,
)
from exp_cascade_sweep import _compa_features, _event_batches  # noqa: E402
from exp_cascade_realgraphs import DATASETS, load_graph, _events as snap_events  # noqa: E402
import exp_cascade_enron as EN  # noqa: E402

SEEDS = [0, 1, 2]
EPOCHS_SNAP = 90
EPOCHS_ENRON = 80
T_BUCKETS_ENRON = EN.N_BUCKETS
WINDOW = 6  # okno cech sondy czasowej (kubełki wstecz)


# ============ czesc wspolna ============


def _gaps(events, label):
    """Odstepy (w kubełkach) miedzy kolejnymi wysylkami tego samego nadawcy.
    Zwraca (lista globalna, lista srednich per nadawca)."""
    per_s = defaultdict(list)
    ev = sorted([(e[2], i) for i, e in enumerate(events) if e[3] == label])
    by_sender = defaultdict(list)
    for _b, i in ev:
        by_sender[events[i][0]].append(events[i][2])
    gaps = []
    sender_means = []
    for s, bs in by_sender.items():
        g = [b2 - b1 for b1, b2 in zip(sorted(bs), sorted(bs)[1:])]
        gaps.extend(g)
        if g:
            sender_means.append(float(np.mean(g)))
    return gaps, sender_means


def _ks_row(graph, regime, events):
    a_g, a_m = _gaps(events, 1)
    b_g, b_m = _gaps(events, 0)
    row = {"graph": graph, "regime": regime}
    for scope, (x, y) in (("global", (a_g, b_g)), ("per_sender", (a_m, b_m))):
        if len(x) >= 5 and len(y) >= 5:
            ks, p = ks_2samp(x, y)
            row[f"ks_{scope}"] = round(float(ks), 4)
            row[f"ks_p_{scope}"] = float(p)
            row[f"gap_attack_{scope}"] = round(float(np.mean(x)), 3)
            row[f"gap_benign_{scope}"] = round(float(np.mean(y)), 3)
    return row


def _probe_features(events, t_buckets):
    """Czysto czasowa sonda: cechy wyłącznie z odstępow/regularności wysylek NADAWCY
    (przyczynowo: tylko wczesniejsze wysylki). Cel: wykryc sygnature iniekcji."""
    n = len(events)
    X = [None] * n
    hist = defaultdict(list)  # sender -> lista bucketow wczesniejszych wysylek
    order = sorted(range(n), key=lambda i: (events[i][2], i))
    for i in order:
        s, v, b, lab = events[i]
        h = hist[s]
        if not h:
            X[i] = [0.0, 0.0, 0.0, float(t_buckets), 1.0]
        else:
            gaps = [b2 - b1 for b1, b2 in zip(h, h[1:])] or [(b - h[-1])]
            last_gap = b - h[-1]
            X[i] = [
                float(last_gap),
                float(np.mean(gaps)),
                float(np.std(gaps)) if len(gaps) > 1 else 0.0,
                float(len(h)),
                1.0 / (1.0 + len(h)),
            ]
        hist[s].append(b)
    return np.array(X, np.float32)


def _victim_split(events, seed):
    vics = sorted({e[1] for e in events})
    rng = np.random.default_rng(seed)
    te_v = set(np.array(vics)[rng.permutation(len(vics))[: max(1, len(vics) // 3)]])
    te = np.array([e[1] in te_v for e in events])
    return te, ~te


def _probe_auc(events, t_buckets, seed):
    X = _probe_features(events, t_buckets)
    y = np.array([e[3] for e in events], dtype=np.float32)
    te, tr = _victim_split(events, seed)
    clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(
        X[tr], y[tr]
    )
    s = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y[te], s))


def _empirical_gaps(events, rng, t_buckets):
    """Empiryczny rozklad odstepow tla: odstepy miedzy kolejnymi wysylkami nadawcow
    benign (modulo T). Zwraca liste odstepow (dla ataku brak historii: globalne)."""
    gaps, _ = _gaps(events, 0)
    return [g % t_buckets for g in gaps] or [1]


def _stealth_empirical(events, rng, t_buckets):
    """Wariant 'stealth-empirical': czasy wysylek ATAKU przelosowane z empirycznego
    rozkladu odstepow tla (per nadawca: przesuwamy kolejne wysylki ataku nadawcy o
    losowane odstepy tla). Benign nietkniety."""
    gap_pool = _empirical_gaps(events, rng, t_buckets)
    out = []
    last_b = defaultdict(int)
    atk = defaultdict(list)
    for e in events:
        if e[3] == 1:
            atk[e[0]].append(e)
        else:
            out.append(e)
    for s, evs in atk.items():
        evs.sort(key=lambda e: e[2])
        b = evs[0][2]
        for k, e in enumerate(evs):
            if k > 0:
                b = (b + int(rng.choice(gap_pool))) % t_buckets
            out.append((e[0], e[1], b, 1))
    rng.shuffle(out)  # ustalony (losowy) porzadek strumienia — jak w generatorach
    return out


def _hand_ctx_features(events, nbr, seed, is_consistent_fn):
    """Kontekst ręczny (framework-1, przyczynowosc zdarzeniowa) dla grafow zewn."""
    n = len(events)
    X = [None] * n
    recv_b = defaultdict(set)
    order = sorted(range(n), key=lambda i: (events[i][2], i))
    W = 2
    for i in order:
        s, v, b, _lab = events[i]
        cs = nbr.get(v, set())
        deg_v = max(1, len(cs))
        recent_recv = set(recv_b.get(b, ()))
        recent_sent = set()
        for bb in range(max(0, b - W), b):
            recent_recv |= recv_b.get(bb, set())
        burst = len((cs - {s}) & recent_recv) / deg_v
        recency = 1.0 if s in recent_recv else 0.0
        X[i] = [
            1.0 if s in cs else 0.0,
            1.0 if is_consistent_fn(s, b, seed) else 0.0,
            float(len(cs)),
            float(len(nbr.get(s, ()))),
            burst,
            recency,
        ]
        recv_b[b].add(v)
    return np.array(X, np.float32)


def _train_tgn(events, idx, n, X1, seed, epochs, t_buckets):
    torch.manual_seed(seed)
    y = np.array([e[3] for e in events], dtype=np.float32)
    te, tr = _victim_split(events, seed)
    recv_b, sent_b = _index_traffic(events)
    Xn = _node_features(events, {}, idx, n, recv_b, sent_b, tr=tr)
    s_idx = torch.tensor([idx[e[0]] for e in events])
    v_idx = torch.tensor([idx[e[1]] for e in events])
    ef = torch.tensor(X1, dtype=torch.float32)
    yt = torch.tensor(y)
    tr_t = torch.tensor(tr)
    te_t = torch.tensor(te)
    m = TemporalGNN(Xn.shape[1], X1.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    lf = nn.BCEWithLogitsLoss()
    batches = _event_batches(events, idx)
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        loss = lf(m(Xn, batches, n, ef, s_idx, v_idx)[tr_t], yt[tr_t])
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        s = torch.sigmoid(m(Xn, batches, n, ef, s_idx, v_idx))[te_t].numpy()
    return roc_auc_score(y[te], s), _recall_at_fpr(y[te], s)


def _eval_panel(events, idx, n, X1, nbr, seed, epochs, t_buckets, is_consistent_fn):
    """COMPA + kontekst ręczny + TGN (+sonda) na danym strumieniu zdarzen."""
    out = {}
    y = np.array([e[3] for e in events], dtype=np.float32)
    te, tr = _victim_split(events, seed)
    out["probe"] = _probe_auc(events, t_buckets, seed)
    for name, X in (
        ("compa", _compa_features(events)),
        ("reczny-kontekst", _hand_ctx_features(events, nbr, seed, is_consistent_fn)),
    ):
        clf = LGBMClassifier(
            n_jobs=1, random_state=42, n_estimators=200, verbose=-1
        ).fit(X[tr], y[tr])
        s = clf.predict_proba(X[te])[:, 1]
        out[name] = (float(roc_auc_score(y[te], s)), _recall_at_fpr(y[te], s))
    out["temporalny-GNN"] = _train_tgn(events, idx, n, X1, seed, epochs, t_buckets)
    return out


def _flat_is_consistent(nbr_unused, s, b, seed):
    return (
        EN.is_consistent(s, b, seed) if hasattr(EN, "is_consistent") else (b % 4 != 3)
    )


# ============ SNAP ============


def snap_run(name, fname, rows, perseed):
    """Detektory dokladnie jak w exp_cascade_realgraphs.run, ale na zadanych zdarzeniach."""
    nbr = load_graph(fname)
    tb = 28

    def snap_eval(ev, seed):
        nodes = sorted({e[0] for e in ev} | {e[1] for e in ev})
        idx = {t: i for i, t in enumerate(nodes)}
        n = len(nodes)
        out = {}
        y = np.array([e[3] for e in ev], dtype=np.float32)
        te, tr = _victim_split(ev, seed)
        recv_b, sent_b = _index_traffic(ev)
        X1, Xc, _y = _features(ev, nbr, recv_b, sent_b, seed)
        Xco = _compa_features(ev)
        for dname, X in (("reczny-kontekst", Xc), ("compa", Xco)):
            clf = LGBMClassifier(
                n_jobs=1, random_state=42, n_estimators=200, verbose=-1
            ).fit(X[tr], y[tr])
            s = clf.predict_proba(X[te])[:, 1]
            out[dname] = (roc_auc_score(y[te], s), _recall_at_fpr(y[te], s))
        Xn = _node_features(ev, nbr, idx, n, recv_b, sent_b, tr=tr)
        s_idx = torch.tensor([idx[e[0]] for e in ev])
        v_idx = torch.tensor([idx[e[1]] for e in ev])
        ef = torch.tensor(X1, dtype=torch.float32)
        yt = torch.tensor(y)
        tr_t = torch.tensor(tr)
        te_t = torch.tensor(te)
        m = TemporalGNN(Xn.shape[1], X1.shape[1])
        torch.manual_seed(seed)
        opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
        lf = nn.BCEWithLogitsLoss()
        batches = _event_batches(ev, idx)
        for _ in range(EPOCHS_SNAP):
            m.train()
            opt.zero_grad()
            loss = lf(m(Xn, batches, n, ef, s_idx, v_idx)[tr_t], yt[tr_t])
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            s = torch.sigmoid(m(Xn, batches, n, ef, s_idx, v_idx))[te_t].numpy()
        out["temporalny-GNN"] = (roc_auc_score(y[te], s), _recall_at_fpr(y[te], s))
        return out

    for seed in SEEDS:
        ev_st = snap_events(nbr, seed, stealth_k=2)
        ev_bl = snap_events(nbr, seed, stealth_k=None)
        ev_em = _stealth_empirical(ev_st, np.random.default_rng(seed), tb)
        for tag, ev in (
            ("blast", ev_bl),
            ("stealthy", ev_st),
            ("stealthy-empirical", ev_em),
        ):
            if seed == 0:
                rows.append({**_ks_row(name, tag, ev)})
            _probe = _probe_auc(ev, tb, seed)
            rows.append(
                {
                    "graph": name,
                    "regime": tag,
                    "seed": seed,
                    "probe_auc": round(_probe, 4),
                }
            )
            if tag in ("stealthy", "stealthy-empirical"):
                res = snap_eval(ev, seed)
                for det, val in res.items():
                    perseed.append(
                        [name, tag, seed, det, round(val[0], 4), round(val[1], 4)]
                    )
                perseed.append([name, tag, seed, "probe", round(_probe, 4), ""])
        print(f"  [{name}] seed {seed} done", flush=True)


# ============ Enron ============


def enron_run(rows, perseed):
    nbr, sent, active = EN.parse_enron()
    tb = EN.N_BUCKETS

    def enron_eval(ev, seed):
        """Detektory dokladnie jak w exp_cascade_enron.run (ten sam protokol)."""
        nodes = sorted({e[0] for e in ev} | {e[1] for e in ev})
        idx = {t: i for i, t in enumerate(nodes)}
        n = len(nodes)
        out = {}
        y = np.array([e[3] for e in ev], dtype=np.float32)
        te, tr = _victim_split(ev, seed)
        recv_b, sent_b = EN._index_traffic(ev)
        X1, Xc = EN._features(ev, nbr, active, recv_b, sent_b, seed)
        Xco = _compa_features(ev)
        for name, X in (("reczny-kontekst", Xc), ("compa", Xco)):
            clf = LGBMClassifier(
                n_jobs=1, random_state=42, n_estimators=200, verbose=-1
            ).fit(X[tr], y[tr])
            s = clf.predict_proba(X[te])[:, 1]
            out[name] = (roc_auc_score(y[te], s), _recall_at_fpr(y[te], s))
        Xn = _node_features(ev, nbr, idx, n, recv_b, sent_b, tr=tr)
        s_idx = torch.tensor([idx[e[0]] for e in ev])
        v_idx = torch.tensor([idx[e[1]] for e in ev])
        ef = torch.tensor(X1, dtype=torch.float32)
        yt = torch.tensor(y)
        tr_t = torch.tensor(tr)
        te_t = torch.tensor(te)
        pb = EN._per_bucket(ev, idx)
        m = EN.TemporalGNN(Xn.shape[1], ef.shape[1])
        torch.manual_seed(seed)
        opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
        lf = nn.BCEWithLogitsLoss()
        for _ in range(EN.EPOCHS):
            m.train()
            opt.zero_grad()
            loss = lf(m(Xn, pb, n, ef, s_idx, v_idx)[tr_t], yt[tr_t])
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            s = torch.sigmoid(m(Xn, pb, n, ef, s_idx, v_idx))[te_t].numpy()
        out["temporalny-GNN"] = (roc_auc_score(y[te], s), _recall_at_fpr(y[te], s))
        return out

    for seed in SEEDS:
        ev_st = EN._events(nbr, sent, seed, stealth_k=2)
        ev_bl = EN._events(nbr, sent, seed, stealth_k=None)
        ev_em = _stealth_empirical(ev_st, np.random.default_rng(seed), tb)
        for tag, ev in (
            ("blast", ev_bl),
            ("stealthy", ev_st),
            ("stealthy-empirical", ev_em),
        ):
            if seed == 0:
                rows.append({**_ks_row("Enron", tag, ev)})
            _probe = _probe_auc(ev, tb, seed)
            rows.append(
                {
                    "graph": "Enron",
                    "regime": tag,
                    "seed": seed,
                    "probe_auc": round(_probe, 4),
                }
            )
            if tag in ("stealthy", "stealthy-empirical"):
                res = enron_eval(ev, seed)
                for det, val in res.items():
                    perseed.append(
                        ["Enron", tag, seed, det, round(val[0], 4), round(val[1], 4)]
                    )
                perseed.append(["Enron", tag, seed, "probe", round(_probe, 4), ""])
        print(f"  [Enron] seed {seed} done", flush=True)


# ============ main ============


def main():
    torch.set_num_threads(4)
    rows, perseed = [], []
    enron_run(rows, perseed)
    for name, fname in DATASETS:
        snap_run(name, fname, rows, perseed)

    out = RESULTS / "exp_injection_probe.csv"
    keys = [
        "graph",
        "regime",
        "seed",
        "probe_auc",
        "ks_global",
        "ks_p_global",
        "ks_per_sender",
        "ks_p_per_sender",
        "gap_attack_global",
        "gap_benign_global",
        "gap_attack_per_sender",
        "gap_benign_per_sender",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    out_ps = RESULTS / "exp_injection_probe_perseed.csv"
    with out_ps.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["graph", "regime", "seed", "detektor", "auc", "r1"])
        w.writerows(perseed)
    print(f"\n[injection-probe] -> {out}\n[injection-probe] -> {out_ps}")


if __name__ == "__main__":
    main()
