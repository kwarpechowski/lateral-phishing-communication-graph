"""Multipleks na REALNYM grafie Enrona + krzywa Recall-FPR (#GP-EXP-13, M1).

Domyka zarzut recenzji M1: dotad na realnym Enronie walidowalismy TYLKO warstwe czasowa.
Tu budujemy z naglowkow REALNY multipleks warstw strukturalnych+czasowych:
  * kontakty   : From -> To  (kto do kogo pisal)
  * wspolodbiorcy : konta wspolwystepujace na liscie To/Cc
  * domena     : ten sam domena adresu
  * rytm czasu : aktywne przedzialy godzin-tygodnia nadawcy (z Date)  [cecha spojnosci]
(OSINT pozostaje syntetyczny -- Enron nie ma takich faktow; to swiadomy podzial.)
Zdarzenia na realnym grafie: nowi nadawcy (off-graph) + przejecia (znany nadawca poza
rytmem). Porownujemy pelny multipleks vs tylko-kontakty i raportujemy AUC oraz Recall przy
WIELU progach FPR (0.1%, 0.5%, 1%, 5%) -- krzywa Recall-FPR (benchmark Ho i in.).

Wyjscie: results/exp_enron_multiplex.csv (Recall przy roznych FPR).
"""

from __future__ import annotations

import csv
import email
import hashlib
from collections import defaultdict
from email.utils import parsedate_to_datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve

ENRON = Path("../../projects/personalized-phishing-defense/code/data/enron/maildir").resolve()
RESULTS = Path(__file__).resolve().parent.parent / "results"
MAX_USERS = 60
MAX_MSGS = 400
MIN_SENT = 25
N_BUCKETS = 28
ACTIVE_MIN = 2
MAX_RECIP = 8                 # pomijaj mass-maile dla warstwy wspolodbiorcow
SEEDS = list(range(5))
FPRS = [0.001, 0.005, 0.01, 0.05]


def _bucket(dt):
    return (dt.weekday() * 4 + dt.hour // 6) % N_BUCKETS


def _h(*p):
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def parse():
    sent = defaultdict(list)        # sender -> [(recipient, bucket)]
    corecip = defaultdict(set)      # account -> set(co-recipients)
    domain = {}                     # account -> domain
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
            recips = [r.strip().lower() for r in to.split(",") if "@" in r]
            for r in recips:
                if r != frm:
                    sent[frm].append((r, b))
            domain.setdefault(frm, frm.split("@")[-1])
            for r in recips:
                domain.setdefault(r, r.split("@")[-1])
            if 2 <= len(recips) <= MAX_RECIP:        # warstwa wspolodbiorcow
                for a, c in combinations(sorted(set(recips)), 2):
                    corecip[a].add(c); corecip[c].add(a)
            n += 1
    sent = {s: v for s, v in sent.items() if len(v) >= MIN_SENT}
    active = {}
    for s, v in sent.items():
        cnt = defaultdict(int)
        for _r, b in v:
            cnt[b] += 1
        active[s] = {b for b, c in cnt.items() if c >= ACTIVE_MIN}
    sent = {s: v for s, v in sent.items() if active.get(s)}
    contact = {s: {r for r, _b in v} for s, v in sent.items()}
    return sent, active, contact, corecip, domain


def _recall_at_fpr(y, s, t):
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(y, s)
    m = fpr <= t
    return float(tpr[m].max()) if m.any() else 0.0


def _events(sent, active, contact, seed):
    senders = sorted(sent)
    sset = set(senders)
    rows = []   # (sender, victim, label, bucket)
    for s in senders:
        act = sorted(active[s])
        off_bkts = sorted(set(range(N_BUCKETS)) - active[s]) or act
        recips = [r for r, _b in sent[s]]
        non_contacts = [x for x in senders if x != s and x not in contact[s]]
        for i, (r, b) in enumerate(sent[s]):
            h = _h(str(seed), s, str(i))
            t = h % 4
            if t == 0:                                  # benign: realna wiadomosc
                rows.append((s, r, 0, b))
            elif t == 1:                                # przejecie: znany odbiorca, poza rytmem
                bb = off_bkts[h % len(off_bkts)] if (h // 5) % 100 >= 25 else act[h % len(act)]
                rows.append((s, r, 1, bb))
            elif t == 2 and non_contacts:               # atak off-graph: obcy nadawca -> r
                rows.append((non_contacts[h % len(non_contacts)], r, 1, b))
            else:                                       # benign: inny realny odbiorca
                rows.append((s, recips[h % len(recips)], 0, b))
    return rows


def _feat(rows, contact, corecip, domain, active, seed, full):
    X = []
    for s, v, _l, b in rows:
        c = 1.0 if (v in contact.get(s, set()) or s in contact.get(v, set())) else 0.0
        if not full:
            X.append([c]); continue
        cc = 1.0 if v in corecip.get(s, set()) else 0.0
        dm = 1.0 if domain.get(s) and domain.get(s) == domain.get(v) else 0.0
        tc = 1.0 if b in active.get(s, set()) else 0.0
        X.append([c, cc, dm, tc, c + cc + dm])
    return np.array(X, np.float32)


def run(sent, active, contact, corecip, domain, seed):
    rows = _events(sent, active, contact, seed)
    y = np.array([r[2] for r in rows]); snd = [r[0] for r in rows]
    order = sorted(set(snd))
    te_s = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order)//3)]])
    te = np.array([x in te_s for x in snd]); tr = ~te

    def fit(full):
        X = _feat(rows, contact, corecip, domain, active, seed, full)
        clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=150, verbose=-1).fit(X[tr], y[tr])
        return clf.predict_proba(X[te])[:, 1]
    yte = y[te]
    out = {}
    for tag, full in (("full", True), ("contact", False)):
        sc = fit(full)
        out[f"{tag}_auc"] = roc_auc_score(yte, sc)
        for fp in FPRS:
            out[f"{tag}_r_{fp}"] = _recall_at_fpr(yte, sc, fp)
    return out, len(order), len(rows)


def main():
    if not ENRON.exists():
        print(f"[enron-mux] brak maildira {ENRON}"); return
    print("[enron-mux] parsowanie (chwile)...", flush=True)
    sent, active, contact, corecip, domain = parse()
    print(f"[enron-mux] nadawcow={len(sent)}, wspolodb. par~{sum(len(v) for v in corecip.values())//2}", flush=True)
    acc = defaultdict(list); ns = nr = 0
    for seed in SEEDS:
        o, ns, nr = run(sent, active, contact, corecip, domain, seed)
        for k, v in o.items():
            acc[k].append(v)
    agg = {k: float(np.mean(v)) for k, v in acc.items()}
    out = RESULTS / "exp_enron_multiplex.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model", "auc"] + [f"recall@fpr{fp}" for fp in FPRS])
        for tag in ("full", "contact"):
            w.writerow([tag, round(agg[f"{tag}_auc"], 4)] + [round(agg[f"{tag}_r_{fp}"], 4) for fp in FPRS])
    print(f"[enron-mux] REALNY multipleks Enron ({ns} nadawcow, ~{nr} zdarzen):")
    print(f"  {'model':8s} {'AUC':>6s} " + " ".join(f"R@{fp:>6}" for fp in FPRS))
    for tag in ("full", "contact"):
        print(f"  {tag:8s} {agg[tag+'_auc']:6.3f} " + " ".join(f"{agg[f'{tag}_r_{fp}']:7.3f}" for fp in FPRS))
    print(f"[enron-mux] wrote {out}")


if __name__ == "__main__":
    main()
