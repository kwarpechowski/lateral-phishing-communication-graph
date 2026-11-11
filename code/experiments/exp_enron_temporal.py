"""Walidacja H8b na REALNYM grafie Enrona: warstwa czasowa vs przejete konta.

Przenosi kluczowy wynik (#GP-EXP-2) z syntetycznej nakladki na realne rytmy komunikacji.
Parsujemy maildir Enrona (From/To/Date), budujemy realny graf kontaktow i realny rytm
aktywnosci nadawcy (przedzialy godzin-tygodnia). Zdarzenia: benign = realna para w realnym
przedziale czasu wiadomosci; przejecie = ta sama znana para, lecz poza rytmem nadawcy.
Cecha statyczna (para znana -> zawsze 1, stad slepota) vs +czas (przedzial w aktywnosci
nadawcy). Wyjscie: results/exp_enron_temporal.csv
"""

from __future__ import annotations

import csv
import email
import hashlib
from collections import defaultdict
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve

ENRON = Path("../../projects/personalized-phishing-defense/code/data/enron/maildir").resolve()
RESULTS = Path(__file__).resolve().parent.parent / "results"
MAX_USERS = 60
MAX_MSGS_PER_USER = 400
MIN_SENT = 25
N_BUCKETS = 28            # 7 dni x 4 przedzialy po 6h
ACTIVE_MIN = 2
SEEDS = list(range(5))


def _bucket(dt) -> int:
    return (dt.weekday() * 4 + dt.hour // 6) % N_BUCKETS


def _h(*p):
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def parse_enron():
    sent = defaultdict(list)
    users = sorted([d for d in ENRON.iterdir() if d.is_dir()])[:MAX_USERS]
    for ud in users:
        n = 0
        for f in ud.rglob("*"):
            if not f.is_file() or n >= MAX_MSGS_PER_USER:
                continue
            try:
                msg = email.message_from_bytes(f.read_bytes())
            except Exception:
                continue
            frm = (msg.get("From") or "").strip().lower()
            to = (msg.get("To") or "")
            date = msg.get("Date")
            if not frm or "@" not in frm or not to or not date:
                continue
            try:
                dt = parsedate_to_datetime(date)
                b = _bucket(dt)
            except Exception:
                continue
            for r in to.split(","):
                r = r.strip().lower()
                if "@" in r and r != frm:
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
    return sent, active


def _recall_at_fpr(y, s, t=0.01):
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(y, s)
    m = fpr <= t
    return float(tpr[m].max()) if m.any() else 0.0


def run(sent, active, seed):
    senders = sorted(sent)
    rows = []
    for s in senders:
        offb = sorted(set(range(N_BUCKETS)) - active[s]) or sorted(active[s])
        actb = sorted(active[s])
        for i, (r, b) in enumerate(sent[s]):
            h = _h(str(seed), s, str(i))
            if h % 2 == 0:
                rows.append((s, r, 0, b))                  # realna wiadomosc, realny bucket
            else:                                          # przejecie; 25% mimikuje rytm
                if (h // 5) % 100 < 25:
                    rows.append((s, r, 1, actb[h % len(actb)]))   # atakujacy w godzinach nadawcy
                else:
                    rows.append((s, r, 1, offb[h % len(offb)]))   # poza rytmem
    y = np.array([x[2] for x in rows])
    snd = [x[0] for x in rows]
    order = sorted(set(snd))
    te_s = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order)//3)]])
    te = np.array([x in te_s for x in snd]); tr = ~te
    Xtemp = np.array([[1.0, 1.0 if b in active[s] else 0.0] for s, _r, _l, b in rows], dtype=np.float32)
    out = {"static_auc": 0.5, "static_r1": 0.0}            # stala cecha -> AUC 0.5 z definicji
    clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(Xtemp[tr], y[tr])
    sc = clf.predict_proba(Xtemp[te])[:, 1]
    out["temporal_auc"] = roc_auc_score(y[te], sc)
    out["temporal_r1"] = _recall_at_fpr(y[te], sc)
    return out, len(senders), len(rows)


def main():
    if not ENRON.exists():
        print(f"[enron] brak maildira: {ENRON}"); return
    print("[enron] parsowanie maildira (chwile)...", flush=True)
    sent, active = parse_enron()
    print(f"[enron] nadawcow z rytmem: {len(sent)}", flush=True)
    acc = defaultdict(list)
    ns = nr = 0
    for seed in SEEDS:
        o, ns, nr = run(sent, active, seed)
        for k, v in o.items():
            acc[k].append(v)
    agg = {k: float(np.mean(v)) for k, v in acc.items()}
    out = RESULTS / "exp_enron_temporal.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["metric", "mean", "n_seeds"])
        for k, v in acc.items():
            w.writerow([k, round(float(np.mean(v)), 4), len(v)])
    print(f"[enron] H8b na REALNYM Enronie ({ns} nadawcow, ~{nr} zdarzen):")
    print(f"  static   AUC={agg['static_auc']:.3f}  R@1%={agg['static_r1']:.3f}")
    print(f"  temporal AUC={agg['temporal_auc']:.3f}  R@1%={agg['temporal_r1']:.3f}")
    print(f"[enron] wrote {out}")


if __name__ == "__main__":
    main()
