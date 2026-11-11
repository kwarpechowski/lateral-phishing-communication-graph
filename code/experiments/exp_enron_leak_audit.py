"""Audyt wycieku rytmu w kotwicy Enron P1 (#GP-EXP-28, recenzja N1).

Podejrzenie (z lekcji P2): flagowy wynik Enron (temporal AUC 0.87, multipleks 0.94) jest zawyzony,
bo benign = realna wiadomosc ZAWSZE w rytmie (tc=1), a atak przejecia poza rytmem (tc=0) -> cecha
czasowa rozdziela trywialnie. Realny ruch JEDNAK bywa off-hours. Test: dodajemy benign off-hours
z rosnacym tempem i patrzymy, jak spada AUC. Jesli mocno spada -> wynik byl artefaktem.

Reuzywa parsera i cech z exp_enron_multiplex. Uruchom z katalogu code/.
Wyjscie: results/exp_enron_leak_audit.csv
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "experiments"))
RESULTS = CODE / "results"

import exp_enron_multiplex as EM                                     # noqa: E402
EM.MAX_USERS = 40                                                    # nieco mniej dla szybkosci
EM.ENRON = CODE.parents[1] / "projects/personalized-phishing-defense/code/data/enron/maildir"  # absolutna sciezka
from exp_enron_multiplex import parse, _feat, _recall_at_fpr, N_BUCKETS, _h  # noqa: E402

SEEDS = list(range(5))
OFF_RATES = [0.0, 0.10, 0.20, 0.30]      # odsetek benign przeniesiony poza rytm
FPR = 0.01


def _events_audit(sent, active, contact, seed, off_rate):
    """Jak EM._events, lecz benign z prawdopod. off_rate trafia POZA rytm nadawcy (off-hours)."""
    senders = sorted(sent)
    rows = []
    for s in senders:
        act = sorted(active[s])
        off_bkts = sorted(set(range(N_BUCKETS)) - active[s]) or act
        recips = [r for r, _b in sent[s]]
        non_contacts = [x for x in senders if x != s and x not in contact[s]]
        for i, (r, b) in enumerate(sent[s]):
            h = _h(str(seed), s, str(i)); t = h % 4

            def benign_bucket(bb, hh):
                # czesc legalnej poczty realnie przychodzi off-hours -> przenies poza rytm
                if (hh // 13) % 100 < off_rate * 100:
                    return off_bkts[hh % len(off_bkts)]
                return bb
            if t == 0:
                rows.append((s, r, 0, benign_bucket(b, h)))
            elif t == 1:
                bb = off_bkts[h % len(off_bkts)] if (h // 5) % 100 >= 25 else act[h % len(act)]
                rows.append((s, r, 1, bb))
            elif t == 2 and non_contacts:
                rows.append((non_contacts[h % len(non_contacts)], r, 1, b))
            else:
                rows.append((s, recips[h % len(recips)], 0, benign_bucket(b, h)))
    return rows


def _tc_only(rows, active):
    return np.array([[1.0 if b in active.get(s, set()) else 0.0] for s, _v, _l, b in rows], np.float32)


def _run(sent, active, contact, corecip, domain, seed, off_rate):
    rows = _events_audit(sent, active, contact, seed, off_rate)
    y = np.array([r[2] for r in rows]); snd = [r[0] for r in rows]
    order = sorted(set(snd))
    te_s = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order) // 3)]])
    te = np.array([x in te_s for x in snd]); tr = ~te
    out = {}
    # tc-only (izoluje wyciek), full multipleks
    feats = {"tc_only": _tc_only(rows, active),
             "full": _feat(rows, contact, corecip, domain, active, seed, True)}
    for tag, X in feats.items():
        clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=150, verbose=-1).fit(X[tr], y[tr])
        sc = clf.predict_proba(X[te])[:, 1]
        out[tag] = (roc_auc_score(y[te], sc), _recall_at_fpr(y[te], sc, FPR))
    return out


def main():
    if not EM.ENRON.exists():
        print(f"[audit] brak maildira {EM.ENRON}"); return
    print("[audit] parsowanie Enrona...", flush=True)
    sent, active, contact, corecip, domain = parse()
    print(f"[audit] nadawcow={len(sent)}", flush=True)
    rows = []
    for off in OFF_RATES:
        acc = {}
        for seed in SEEDS:
            r = _run(sent, active, contact, corecip, domain, seed, off)
            for k, (a, rr) in r.items():
                acc.setdefault(k, {"a": [], "r": []})
                acc[k]["a"].append(a); acc[k]["r"].append(rr)
        line = " | ".join(f"{k} auc={np.mean(v['a']):.3f} r@1%={np.mean(v['r']):.3f}" for k, v in acc.items())
        print(f"  benign off-hours={off:.0%}: {line}", flush=True)
        for k, v in acc.items():
            rows.append([off, k, round(float(np.mean(v["a"])), 4), round(float(np.mean(v["r"])), 4)])
    out = RESULTS / "exp_enron_leak_audit.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["off_hours_rate", "model", "auc", "recall_fpr1"])
        w.writerows(rows)
    print(f"\n[audit] -> {out}")


if __name__ == "__main__":
    main()
