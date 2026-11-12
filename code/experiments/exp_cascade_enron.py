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
ENRON = (
    Path(__file__).resolve().parents[3]
    / "projects/personalized-phishing-defense/code/data/enron/maildir"
)

from exp_cascade import TemporalGNN, _index_traffic, _node_features, _recall_at_fpr  # noqa: E402
from exp_cascade_sweep import _compa_features  # noqa: E402

MAX_USERS = 30
MAX_MSGS = 300
MIN_SENT = 20
N_BUCKETS = 28
ACTIVE_MIN = 2
CORE_MIN_DEG = 2  # odetnij jednorazowych odbiorcow (stopien <2)
CORE_CAP = 2000  # top-K kont po stopniu (aktywny rdzen, by GNN byl wykonalny na CPU)
SEEDS = list(range(8))
N_CASCADES = 25
MAX_HOPS = 3
P_INFECT = 0.6
MAX_FANOUT = (
    10  # max kontaktow zaatakowanych na nosiciela (realistyczne + ogranicza eksplozje
)
# na gestym realnym grafie; bez tego kaskada generuje dziesiatki tys. zdarzen)
WINDOW = 2
BENIGN_RATIO = 3  # benign : atak
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
            nbr_full[s].add(r)
            nbr_full[r].add(s)
    # AKTYWNY RDZEN: konta o stopniu >= CORE_MIN_DEG, cap top-K po stopniu (wykonalnosc GNN na CPU)
    deg = {a: len(ns) for a, ns in nbr_full.items() if len(ns) >= CORE_MIN_DEG}
    core = set(sorted(deg, key=deg.get, reverse=True)[:CORE_CAP])
    nbr = {a: (nbr_full[a] & core) for a in core}
    nbr = {a: ns for a, ns in nbr.items() if ns}  # usun izolowane po przycieciu
    core = set(nbr)
    # przytnij sent/active do rdzenia (nadawca i odbiorca w rdzeniu)
    sent = {s: [(r, b) for (r, b) in v if r in core] for s in sent if s in core}
    sent = {s: v for s, v in sent.items() if len(v) >= ACTIVE_MIN and active.get(s)}
    active = {s: active[s] for s in sent}
    return nbr, sent, active


def _benign_gap_pool(edges, seed):
    """R4-A2: empiryczny rozklad odstepow tla Enrona — odstepy miedzy kolejnymi
    wysylaniami tego samego nadawcy w matched benign (kubelki jednostajne)."""
    rng = np.random.default_rng(seed + 555)
    by_sender = {}
    for _ in range(3000):
        u, w = edges[int(rng.integers(len(edges)))]
        b = int(rng.integers(N_BUCKETS))
        by_sender.setdefault(u, []).append(b)
    pool = []
    for bs in by_sender.values():
        bs.sort()
        pool.extend(b2 - b1 for b1, b2 in zip(bs, bs[1:]) if b2 > b1)
    return pool or [1]


def _events(nbr, sent, seed, stealth_k=None, timing="det"):
    """Realne benign + wstrzykniete kaskady.

    stealth_k=None -> naiwny blast (fanout<=MAX_FANOUT na hop). stealth_k=K ->
    stealthy: <=K wyslan/nosiciela, PO JEDNYM na bucket (brak burstu nadawczego)."""
    """Realne benign + wstrzykniete kaskady. Zwraca lista (s, v, bucket, label)."""
    rng = np.random.default_rng(seed)
    senders = sorted(sent)
    edges_all = [(u, w) for u in nbr for w in nbr[u]]
    # --- kaskady na realnym grafie ---
    casc = []
    seeds0 = [senders[i] for i in rng.permutation(len(senders))[:N_CASCADES]]
    for s0 in seeds0:
        t0 = int(rng.integers(0, N_BUCKETS - MAX_HOPS))
        if stealth_k is None:  # BLAST
            infected = {s0}
            frontier = [s0]
            for hop in range(MAX_HOPS):
                nxt = []
                bucket = (t0 + hop) % N_BUCKETS
                for u in frontier:
                    tg = sorted(nbr.get(u, ()))
                    rng.shuffle(tg)
                    for v in tg[:MAX_FANOUT]:  # cap rozgalezienia
                        if v in infected:
                            continue
                        casc.append((u, v, bucket, 1))
                        if rng.random() < P_INFECT:
                            infected.add(v)
                            nxt.append(v)
                frontier = nxt
                if not frontier:
                    break
        else:  # STEALTHY (1 wyslanie / bucket; emp: odstepy z rozkladu tla)
            gap_pool = _benign_gap_pool(edges_all, seed) if timing == "emp" else None
            infected = {s0}
            frontier = [(s0, t0)]
            for _hop in range(MAX_HOPS):
                nxt = []
                for u, tu in frontier:
                    tg = sorted(nbr.get(u, ()))
                    rng.shuffle(tg)
                    if timing == "emp" and gap_pool:
                        gaps = [0] + [
                            gap_pool[int(rng.integers(len(gap_pool)))]
                            for _ in range(stealth_k)
                        ]
                        cum = np.cumsum(gaps)
                    sent_n = 0
                    for v in tg:
                        if v in infected or sent_n >= stealth_k:
                            continue
                        if timing == "emp" and gap_pool:
                            bucket = (tu + int(cum[sent_n])) % N_BUCKETS
                        else:
                            bucket = (tu + 1 + sent_n) % N_BUCKETS
                        casc.append((u, v, bucket, 1))
                        sent_n += 1
                        if rng.random() < P_INFECT:
                            infected.add(v)
                            nxt.append((v, bucket))
                frontier = nxt
                if not frontier:
                    break
    # --- benign DOPASOWANY do ataku (usuwa konfundent stopnia/tozsamosci) ---
    # losowe krawedzie kontaktu (rozklad koncow i stopni jak przy traversal kaskady, bo obie
    # operuja na tym samym zbiorze krawedzi), czas ROZPROSZONY uniformnie. Benign vs atak rozni
    # TYLKO czasowy wzorzec: rozproszenie vs zsynchronizowany burst propagacji. Graf = REALNY Enron.
    benign = []
    for _ in range(BENIGN_RATIO * len(casc)):
        u, w = edges_all[int(rng.integers(len(edges_all)))]
        benign.append((u, w, int(rng.integers(N_BUCKETS)), 0))
    events = casc + benign
    rng.shuffle(events)
    return events


def _features(events, nbr, active, recv_b, sent_b, seed):
    """X_1hop, X_ctx (1hop + kontekst kaskady). Rytm = REALNE aktywne przedzialy nadawcy.

    PRZYCZYNOWOSC ZDARZENIOWA (jak framework-1): zdarzenie e widzi wylacznie zdarzenia
    o scisle wczesniejszym t(e) = (kubelek, pozycja na tasowanej liscie) — kompletne
    kubełki [b-W, b-1] ORAZ prefix biezacego kubełka (regula remisow: porzadek listy)."""
    n = len(events)
    X1 = [None] * n
    Xc = [None] * n
    r_pref: dict = defaultdict(set)  # kubelek -> odbiorcy (prefix biezacego kubełka)
    s_pref: dict = defaultdict(set)  # kubelek -> nadawcy (prefix)
    order = sorted(range(n), key=lambda i: (events[i][2], i))
    for i in order:
        s, v, b, _lab = events[i]
        cs = nbr.get(v, set())
        deg_v, deg_s = len(cs), len(nbr.get(s, set()))
        # UWAGA: bez bitu rytmu (b in active[s]) -- to wyciek: realny benign jest ZAWSZE w rytmie,
        # a syntetyczna kaskada nie, wiec rytm trywialnie rozdziela. Sygnal kaskady = kontekst propagacji.
        f1 = [1.0 if s in cs else 0.0, float(deg_v), float(deg_s)]
        recent_recv = set(r_pref.get(b, ()))
        recent_sent = set(s_pref.get(b, ()))
        for bb in range(max(0, b - WINDOW), b):
            recent_recv |= recv_b.get(bb, set())
            recent_sent |= sent_b.get(bb, set())
        burst = len((cs - {s}) & recent_recv) / max(1, deg_v)
        sender_recency = 1.0 if s in recent_recv else 0.0
        prop = len((cs - {s}) & recent_sent) / max(1, deg_v)
        Xc[i] = f1 + [burst, sender_recency, prop]
        X1[i] = f1
        r_pref[b].add(v)
        s_pref[b].add(s)
    return (
        np.array(X1, np.float32),
        np.array(Xc, np.float32),
    )


def _per_bucket(events, idx, chunk=64):
    """Mini-batche zdarzen w porzadku (kubelek, pozycja) — przyczynowosc zdarzeniowa
    (fix po recenzji wewnetrznej): batch nie przekracza kubełka, batch jest wspolczesny."""
    by_b = {}
    for i, e in enumerate(events):
        by_b.setdefault(e[2], []).append(i)
    pb = []
    for b in sorted(by_b):
        g = by_b[b]
        for c in range(0, len(g), chunk):
            part = g[c : c + chunk]
            sb = torch.tensor([idx[events[i][0]] for i in part])
            vb = torch.tensor([idx[events[i][1]] for i in part])
            pb.append((b, sb, vb, part))
    return pb


def _train_temporal(Xn, pb, n, ef, s_idx, v_idx, yt, tr_t, te_t, y, te, seed):
    torch.manual_seed(seed)
    m = TemporalGNN(Xn.shape[1], ef.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        m.train()
        opt.zero_grad()
        loss = lossf(m(Xn, pb, n, ef, s_idx, v_idx)[tr_t], yt[tr_t])
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        s = torch.sigmoid(m(Xn, pb, n, ef, s_idx, v_idx))[te_t].numpy()
    return roc_auc_score(y[te], s), _recall_at_fpr(y[te], s)


def _timing_stats(events):
    """R4-A2: KS odstepow atak vs tlo + sonda czysto czasowa (per seed, in-run)."""
    from scipy.stats import ks_2samp

    from exp_injection_probe import _gaps, _probe_auc

    a_g, _ = _gaps(events, 1)
    b_g, _ = _gaps(events, 0)
    if len(a_g) >= 5 and len(b_g) >= 5:
        ks, p = ks_2samp(a_g, b_g)
        ks = float(ks)
    else:
        ks, p = float("nan"), float("nan")
    probe = _probe_auc(events, N_BUCKETS, 123)
    return ks, float(p), probe


def run(nbr, sent, active, seed, stealth_k=None, timing="det"):
    events = _events(nbr, sent, seed, stealth_k, timing)
    nodes = sorted({e[0] for e in events} | {e[1] for e in events})
    idx = {t: i for i, t in enumerate(nodes)}
    n = len(nodes)
    recv_b, sent_b = _index_traffic(events)
    X1, Xc = _features(events, nbr, active, recv_b, sent_b, seed)
    y = np.array([e[3] for e in events], dtype=np.float32)
    # split po ofiarach
    vics = sorted({e[1] for e in events})
    rng = np.random.default_rng(seed)
    te_v = set(np.array(vics)[rng.permutation(len(vics))[: max(1, len(vics) // 3)]])
    te = np.array([e[1] in te_v for e in events])
    tr = ~te
    out = {}
    c1 = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(
        X1[tr], y[tr]
    )
    s1 = c1.predict_proba(X1[te])[:, 1]
    out["tab_1hop"] = (roc_auc_score(y[te], s1), _recall_at_fpr(y[te], s1))
    cc = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(
        Xc[tr], y[tr]
    )
    sc = cc.predict_proba(Xc[te])[:, 1]
    out["tab_ctx"] = (roc_auc_score(y[te], sc), _recall_at_fpr(y[te], sc))
    # A5: detektor wolumetryczny (COMPA, przyczynowosc zdarzeniowa) w obu rezymach
    Xco = _compa_features(events)
    cv = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(
        Xco[tr], y[tr]
    )
    sv = cv.predict_proba(Xco[te])[:, 1]
    out["compa"] = (roc_auc_score(y[te], sv), _recall_at_fpr(y[te], sv))
    # temporalny GNN
    Xn = _node_features(events, nbr, idx, n, recv_b, sent_b, tr=tr)
    s_idx = torch.tensor([idx[e[0]] for e in events])
    v_idx = torch.tensor([idx[e[1]] for e in events])
    ef = torch.tensor(X1, dtype=torch.float32)
    yt = torch.tensor(y)
    tr_t = torch.tensor(tr)
    te_t = torch.tensor(te)
    pb = _per_bucket(events, idx)
    out["gnn_temporal"] = _train_temporal(
        Xn, pb, n, ef, s_idx, v_idx, yt, tr_t, te_t, y, te, seed
    )
    # kontrola: przetasowany czas
    shuf = rng.integers(0, N_BUCKETS, size=len(events))
    ev_shuf = [
        (events[i][0], events[i][1], int(shuf[i]), events[i][3])
        for i in range(len(events))
    ]
    pb_s = _per_bucket(ev_shuf, idx)
    out["gnn_temporal_shuf"] = _train_temporal(
        Xn, pb_s, n, ef, s_idx, v_idx, yt, tr_t, te_t, y, te, seed
    )
    ks, kp, probe = _timing_stats(events)
    return out, n, len(events), int(y.sum()), ks, probe


def main():
    if not ENRON.exists():
        print(f"[enron-casc] brak maildira {ENRON}")
        return
    print("[enron-casc] parsowanie realnego Enrona...", flush=True)
    nbr, sent, active = parse_enron()
    print(f"[enron-casc] nadawcow={len(sent)}, wezlow-kontaktu={len(nbr)}", flush=True)
    agg = {}
    perseed = []
    trows = []
    nn_ = ne = na = 0
    # R4-A2: blast det; stealthy w obu wariantach (det/emp)
    # R4: opcjonalny filtr rezymu (rownoleglosc: --rezym blast / stealthy)
    only = None
    if "--rezym" in sys.argv:
        only = sys.argv[sys.argv.index("--rezym") + 1]
    variants = [
        ("blast", None, "det"),
        ("stealthy", 2, "det"),
        ("stealthy", 2, "emp"),
    ]
    for regime, sk, timing in variants:
        if only and regime != only:
            continue
        print(f"=== [enron-casc] rezym: {regime}/{timing} ===", flush=True)
        ks_l, pr_l = [], []
        for seed in SEEDS:
            res, nn_, ne, na, ks, probe = run(
                nbr, sent, active, seed, stealth_k=sk, timing=timing
            )
            ks_l.append(ks)
            pr_l.append(probe)
            for k, (a, r) in res.items():
                agg.setdefault((regime, timing, k), {"auc": [], "r1": []})
                agg[(regime, timing, k)]["auc"].append(a)
                agg[(regime, timing, k)]["r1"].append(r)
                perseed.append([regime, timing, seed, k, round(a, 4), round(r, 4)])
            line = " | ".join(
                f"{k} auc={v[0]:.3f} r1={v[1]:.3f}" for k, v in res.items()
            )
            print(
                f"  [{regime}/{timing}] seed={seed} (wezly {nn_}, zdarz {ne}, ataki {na}):"
                f" {line} | KS={ks:.2f} probe={probe:.2f}",
                flush=True,
            )
        trows.append(
            [
                regime,
                timing,
                round(float(np.nanmean(ks_l)), 4),
                round(float(np.mean(pr_l)), 4),
            ]
        )
    suffix = f"_{only}" if only else ""
    out = RESULTS / f"exp_cascade_enron{suffix}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rezym", "timing", "model", "auc", "recall_fpr1", "n_seeds"])
        for (regime, timing, k), v in agg.items():
            w.writerow(
                [
                    regime,
                    timing,
                    k,
                    round(float(np.mean(v["auc"])), 4),
                    round(float(np.mean(v["r1"])), 4),
                    len(v["auc"]),
                ]
            )
    suffix = f"_{only}" if only else ""
    out_t = RESULTS / f"exp_cascade_enron_timing{suffix}.csv"
    with out_t.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rezym", "timing", "ks_mean", "probe_auc"])
        w.writerows(trows)
    print(f"[enron-timing] -> {out_t}")
    out_ps = RESULTS / f"exp_cascade_enron{suffix}_perseed.csv"
    with out_ps.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rezym", "timing", "seed", "model", "auc", "recall_fpr1"])
        w.writerows(perseed)
    print(f"\n[enron-casc] REALNY graf Enrona -> {out}")
    for (regime, k), v in agg.items():
        if k in ("tab_1hop", "tab_ctx", "compa", "gnn_temporal", "gnn_temporal_shuf"):
            print(
                f"  [{regime}] {k}: AUC={np.mean(v['auc']):.3f}  Recall@FPR1%={np.mean(v['r1']):.3f}"
            )


if __name__ == "__main__":
    main()
