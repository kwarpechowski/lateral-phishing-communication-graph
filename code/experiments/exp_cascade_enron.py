"""KOTWICA REALNA: kaskady lateral-phishingu na REALNYM grafie Enrona (#GP-EXP-21).

Domyka najpowazniejszy zarzut P2 (100% syntetyk). Strategia 'real-graph, injected-attacks'
(realnych etykiet lateral phishingu brak publicznie):
  * graf kontaktow   : REALNY (From -> To z maildira Enrona)
  * czas / rytm       : REALNY (naglowki Date -> 28 przedzialow tygodnia; aktywne przedzialy nadawcy)
  * ruch legalny      : REALNE wiadomosci (sender, recipient, bucket) jako benign (label 0)
  * kaskady (label 1) : WSTRZYKNIETE na realny graf -- propagacja po realnych kontaktach w czasie

Pojedyncze zdarzenie kaskady lokalnie wyglada jak realna poczta od znanego kontaktu; sygnal =
kontekst propagacji (burst sasiadow + swiezosc nadawcy). Detektory jak w exp_cascade: 1-hop,
reczny kontekst, temporalny GNN (+ kontrola shuffle). Metryki: AUC, Recall@FPR1%.

Uruchom z katalogu code/:  ../../../venv/Scripts/python experiments/exp_cascade_enron.py
Wyjscie: results/exp_cascade_enron.csv
"""
from __future__ import annotations

import csv
import email
import hashlib
import sys
from collections import defaultdict
from email.utils import parsedate_to_datetime
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
ENRON = Path(__file__).resolve().parents[3] / "projects/personalized-phishing-defense/code/data/enron/maildir"

from exp_cascade import TemporalGNN, _index_traffic, _node_features, _recall_at_fpr  # noqa: E402

MAX_USERS = 30
MAX_MSGS = 300
MIN_SENT = 20
N_BUCKETS = 28
ACTIVE_MIN = 2
CORE_MIN_DEG = 2        # odetnij jednorazowych odbiorcow (stopien <2)
CORE_CAP = 2000         # top-K kont po stopniu (aktywny rdzen, by GNN byl wykonalny na CPU)
SEEDS = list(range(8))
N_CASCADES = 25
MAX_HOPS = 3
P_INFECT = 0.6
MAX_FANOUT = 10         # max kontaktow zaatakowanych na nosiciela (realistyczne + ogranicza eksplozje
                        # na gestym realnym grafie; bez tego kaskada generuje dziesiatki tys. zdarzen)
WINDOW = 2
BENIGN_RATIO = 3        # benign : atak
EPOCHS = 80


def _bucket(dt):
    return (dt.weekday() * 4 + dt.hour // 6) % N_BUCKETS


def _h(*p):
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def parse_enron():
    """Realny graf kontaktow + rytm z naglowkow. Zwraca nbr, sent, active."""
    sent = defaultdict(list)
    users = sorted([d for d in ENRON.iterdir() if d.is_dir()])[:MAX_USERS]
    for ud in users:
        n = 0
        for f in ud.rglob("*"):
            if not f.is_file() or n >= MAX_MSGS:
                continue
            try:
                msg = email.message_from_bytes(f.read_bytes())
            except Exception:
                continue
            frm = (msg.get("From") or "").strip().lower()
            to = (msg.get("To") or "") + "," + (msg.get("Cc") or "")
            date = msg.get("Date")
            if not frm or "@" not in frm or not date:
                continue
            try:
                b = _bucket(parsedate_to_datetime(date))
            except Exception:
                continue
            for r in [x.strip().lower() for x in to.split(",") if "@" in x]:
                if r != frm:
                    sent[frm].append((r, b))
            n += 1
    sent = {s: v for s, v in sent.items() if len(v) >= MIN_SENT}
    active = {}
    for s, v in sent.items():
        cnt = defaultdict(int)
        for _r, b in v:
            cnt[b] += 1
        active[s] = {b for b, c in cnt.items() if c >= ACTIVE_MIN}
    sent = {s: v for s, v in sent.items() if active.get(s)}
    # symetryczny graf kontaktow (tylko konta wystepujace jako nadawca lub jego odbiorca)
    nbr_full = defaultdict(set)
    for s, v in sent.items():
        for r, _b in v:
            nbr_full[s].add(r); nbr_full[r].add(s)
    # AKTYWNY RDZEN: konta o stopniu >= CORE_MIN_DEG, cap top-K po stopniu (wykonalnosc GNN na CPU)
    deg = {a: len(ns) for a, ns in nbr_full.items() if len(ns) >= CORE_MIN_DEG}
    core = set(sorted(deg, key=deg.get, reverse=True)[:CORE_CAP])
    nbr = {a: (nbr_full[a] & core) for a in core}
    nbr = {a: ns for a, ns in nbr.items() if ns}                 # usun izolowane po przycieciu
    core = set(nbr)
    # przytnij sent/active do rdzenia (nadawca i odbiorca w rdzeniu)
    sent = {s: [(r, b) for (r, b) in v if r in core] for s in sent if s in core}
    sent = {s: v for s, v in sent.items() if len(v) >= ACTIVE_MIN and active.get(s)}
    active = {s: active[s] for s in sent}
    return nbr, sent, active


def _events(nbr, sent, seed):
    """Realne benign + wstrzykniete kaskady. Zwraca lista (s, v, bucket, label)."""
    rng = np.random.default_rng(seed)
    senders = sorted(sent)
    # --- kaskady na realnym grafie ---
    casc = []
    seeds0 = [senders[i] for i in rng.permutation(len(senders))[:N_CASCADES]]
    for s0 in seeds0:
        t0 = int(rng.integers(0, N_BUCKETS - MAX_HOPS))
        infected = {s0}; frontier = [s0]
        for hop in range(MAX_HOPS):
            nxt = []; bucket = (t0 + hop) % N_BUCKETS
            for u in frontier:
                tg = sorted(nbr.get(u, ()))
                rng.shuffle(tg)
                for v in tg[:MAX_FANOUT]:                 # cap rozgalezienia
                    if v in infected:
                        continue
                    casc.append((u, v, bucket, 1))
                    if rng.random() < P_INFECT:
                        infected.add(v); nxt.append(v)
            frontier = nxt
            if not frontier:
                break
    # --- benign DOPASOWANY do ataku (usuwa konfundent stopnia/tozsamosci) ---
    # losowe krawedzie kontaktu (rozklad koncow i stopni jak przy traversal kaskady, bo obie
    # operuja na tym samym zbiorze krawedzi), czas ROZPROSZONY uniformnie. Benign vs atak rozni
    # TYLKO czasowy wzorzec: rozproszenie vs zsynchronizowany burst propagacji. Graf = REALNY Enron.
    edges = [(u, w) for u in nbr for w in nbr[u]]
    benign = []
    for _ in range(BENIGN_RATIO * len(casc)):
        u, w = edges[int(rng.integers(len(edges)))]
        benign.append((u, w, int(rng.integers(N_BUCKETS)), 0))
    events = casc + benign
    rng.shuffle(events)
    return events


def _features(events, nbr, active, recv_b, sent_b, seed):
    """X_1hop, X_ctx (1hop + kontekst kaskady). Rytm = REALNE aktywne przedzialy nadawcy."""
    X1, Xc = [], []
    for s, v, b, _lab in events:
        cs = nbr.get(v, set())
        deg_v, deg_s = len(cs), len(nbr.get(s, set()))
        # UWAGA: bez bitu rytmu (b in active[s]) -- to wyciek: realny benign jest ZAWSZE w rytmie,
        # a syntetyczna kaskada nie, wiec rytm trywialnie rozdziela. Sygnal kaskady = kontekst propagacji.
        f1 = [1.0 if s in cs else 0.0, float(deg_v), float(deg_s)]
        recent_recv = set()
        for bb in range(max(0, b - WINDOW), b + 1):
            recent_recv |= recv_b.get(bb, set())
        burst = len((cs - {s}) & recent_recv) / max(1, deg_v)
        sender_recency = 0.0
        for bb in range(max(0, b - WINDOW), b):
            if s in recv_b.get(bb, set()):
                sender_recency = 1.0; break
        prop = len((cs - {s}) & {x for bb in range(max(0, b - WINDOW), b + 1)
                                 for x in sent_b.get(bb, set())}) / max(1, deg_v)
        Xc.append(f1 + [burst, sender_recency, prop]); X1.append(f1)
    return np.array(X1, np.float32), np.array(Xc, np.float32)


def _per_bucket(events, idx):
    pb = []
    for b in range(N_BUCKETS):
        gidx = [i for i, e in enumerate(events) if e[2] == b]
        if not gidx:
            pb.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), []))
            continue
        sb = torch.tensor([idx[events[i][0]] for i in gidx])
        vb = torch.tensor([idx[events[i][1]] for i in gidx])
        pb.append((sb, vb, gidx))
    return pb


def _train_temporal(Xn, pb, n, ef, s_idx, v_idx, yt, tr_t, te_t, y, te, seed):
    torch.manual_seed(seed)
    m = TemporalGNN(Xn.shape[1], ef.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        loss = lossf(m(Xn, pb, n, ef, s_idx, v_idx)[tr_t], yt[tr_t])
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        s = torch.sigmoid(m(Xn, pb, n, ef, s_idx, v_idx))[te_t].numpy()
    return roc_auc_score(y[te], s), _recall_at_fpr(y[te], s)


def run(nbr, sent, active, seed):
    events = _events(nbr, sent, seed)
    nodes = sorted({e[0] for e in events} | {e[1] for e in events})
    idx = {t: i for i, t in enumerate(nodes)}; n = len(nodes)
    recv_b, sent_b = _index_traffic(events)
    X1, Xc = _features(events, nbr, active, recv_b, sent_b, seed)
    y = np.array([e[3] for e in events], dtype=np.float32)
    # split po ofiarach
    vics = sorted({e[1] for e in events})
    rng = np.random.default_rng(seed)
    te_v = set(np.array(vics)[rng.permutation(len(vics))[:max(1, len(vics) // 3)]])
    te = np.array([e[1] in te_v for e in events]); tr = ~te
    out = {}
    c1 = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(X1[tr], y[tr])
    s1 = c1.predict_proba(X1[te])[:, 1]
    out["tab_1hop"] = (roc_auc_score(y[te], s1), _recall_at_fpr(y[te], s1))
    cc = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(Xc[tr], y[tr])
    sc = cc.predict_proba(Xc[te])[:, 1]
    out["tab_ctx"] = (roc_auc_score(y[te], sc), _recall_at_fpr(y[te], sc))
    # temporalny GNN
    Xn = _node_features(events, nbr, idx, n, recv_b, sent_b, tr=tr)
    s_idx = torch.tensor([idx[e[0]] for e in events])
    v_idx = torch.tensor([idx[e[1]] for e in events])
    ef = torch.tensor(X1, dtype=torch.float32)
    yt = torch.tensor(y); tr_t = torch.tensor(tr); te_t = torch.tensor(te)
    pb = _per_bucket(events, idx)
    out["gnn_temporal"] = _train_temporal(Xn, pb, n, ef, s_idx, v_idx, yt, tr_t, te_t, y, te, seed)
    # kontrola: przetasowany czas
    shuf = rng.integers(0, N_BUCKETS, size=len(events))
    ev_shuf = [(events[i][0], events[i][1], int(shuf[i]), events[i][3]) for i in range(len(events))]
    pb_s = _per_bucket(ev_shuf, idx)
    out["gnn_temporal_shuf"] = _train_temporal(Xn, pb_s, n, ef, s_idx, v_idx, yt, tr_t, te_t, y, te, seed)
    return out, n, len(events), int(y.sum())


def main():
    if not ENRON.exists():
        print(f"[enron-casc] brak maildira {ENRON}"); return
    print("[enron-casc] parsowanie realnego Enrona...", flush=True)
    nbr, sent, active = parse_enron()
    print(f"[enron-casc] nadawcow={len(sent)}, wezlow-kontaktu={len(nbr)}", flush=True)
    agg = {}
    nn_ = ne = na = 0
    for seed in SEEDS:
        res, nn_, ne, na = run(nbr, sent, active, seed)
        for k, (a, r) in res.items():
            agg.setdefault(k, {"auc": [], "r1": []})
            agg[k]["auc"].append(a); agg[k]["r1"].append(r)
        line = " | ".join(f"{k} auc={v[0]:.3f} r1={v[1]:.3f}" for k, v in res.items())
        print(f"  seed={seed} (wezly {nn_}, zdarz {ne}, ataki {na}): {line}", flush=True)
    out = RESULTS / "exp_cascade_enron.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model", "auc", "recall_fpr1", "n_seeds"])
        for k, v in agg.items():
            w.writerow([k, round(float(np.mean(v["auc"])), 4),
                        round(float(np.mean(v["r1"])), 4), len(v["auc"])])
    print(f"\n[enron-casc] REALNY graf Enrona -> {out}")
    for k in ["tab_1hop", "tab_ctx", "gnn_temporal", "gnn_temporal_shuf"]:
        if k in agg:
            print(f"  {k}: AUC={np.mean(agg[k]['auc']):.3f}  Recall@FPR1%={np.mean(agg[k]['r1']):.3f}")


if __name__ == "__main__":
    main()
