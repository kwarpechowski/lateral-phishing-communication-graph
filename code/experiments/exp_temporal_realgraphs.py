"""Walidacja warstwy czasowej P1 na WIELU realnych grafach, leak-aware (#GP-EXP-29, recenzja n3/n1).

EU-email i CollegeMsg (SNAP) maja REALNE timestampy -> realny rytm aktywnosci nadawcy (jak Enron,
ale dwa dodatkowe grafy). Test warstwy spojnosci czasowej (przejecie = znany nadawca poza rytmem)
z LEAK-AWARE projektem: benign dostaje realistyczny odsetek off-hours (15%), by uniknac wycieku
zlapanego w audycie EXP-28. + test istotnosci (Wilcoxon, temporal vs static) na ziarnach.

Wyjscie: results/exp_temporal_realgraphs.csv
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import wilcoxon

CODE = Path(__file__).resolve().parent.parent
RESULTS = CODE / "results"
GRAPHS = CODE / "data" / "realgraphs"

N_BUCKETS = 28
BUCKET_SECONDS = 21600          # 6h -> 28 przedzialow tygodnia
MIN_SENT = 20
ACTIVE_MIN = 2
SEEDS = list(range(20))
OFF_RATE = 0.15                 # realistyczny odsetek benign off-hours (leak-aware)
MIMIC = 0.25                    # odsetek atakow mimikujacych rytm
FPR = 0.01
DATASETS = [("email-Eu-core", "eu.txt"), ("CollegeMsg", "college.txt")]


def _h(*p):
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def _recall_at_fpr(y, s, t=FPR):
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(y, s)
    m = fpr <= t
    return float(tpr[m].max()) if m.any() else 0.0


def load(fname):
    sent = defaultdict(list)
    with (GRAPHS / fname).open() as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            u, v, ts = p[0], p[1], int(p[2])
            if u != v:
                sent[u].append((v, (ts // BUCKET_SECONDS) % N_BUCKETS))
    sent = {s: v for s, v in sent.items() if len(v) >= MIN_SENT}
    active = {}
    for s, v in sent.items():
        cnt = Counter(b for _r, b in v)
        active[s] = {b for b, c in cnt.items() if c >= ACTIVE_MIN}
    sent = {s: v for s, v in sent.items() if active.get(s)}
    contact = {s: {r for r, _b in v} for s, v in sent.items()}
    return sent, active, contact


def _events(sent, active, seed, off_rate):
    """Benign (realne, z off-hours) + przejecia (znany nadawca poza rytmem, 25% mimikry)."""
    rows = []
    for s in sorted(sent):
        act = sorted(active[s])
        off = sorted(set(range(N_BUCKETS)) - active[s]) or act
        recips = [r for r, _b in sent[s]]
        for i, (r, b) in enumerate(sent[s]):
            h = _h(str(seed), s, str(i)); t = h % 2
            if t == 0:                                          # benign realny (+off-hours)
                bb = off[h % len(off)] if (h // 13) % 100 < off_rate * 100 else b
                rows.append((s, r, 0, bb))
            else:                                               # przejecie: znany nadawca poza rytmem
                bb = act[h % len(act)] if (h // 5) % 100 < MIMIC * 100 else off[h % len(off)]
                rr = recips[h % len(recips)]
                rows.append((s, rr, 1, bb))
    return rows


def run(sent, active, seed, off_rate):
    rows = _events(sent, active, seed, off_rate)
    y = np.array([r[2] for r in rows]); snd = [r[0] for r in rows]
    order = sorted(set(snd))
    te_s = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order) // 3)]])
    te = np.array([x in te_s for x in snd]); tr = ~te
    Xst = np.ones((len(rows), 1), np.float32)                  # static: znana para -> 0.5 z konstrukcji
    Xtc = np.array([[1.0 if b in active.get(s, set()) else 0.0] for s, _v, _l, b in rows], np.float32)
    out = {}
    for tag, X in (("static", Xst), ("temporal", Xtc)):
        if tag == "static":
            out[tag] = (0.5, 0.0)                               # stala cecha -> brak separacji
            continue
        clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=120, verbose=-1).fit(X[tr], y[tr])
        sc = clf.predict_proba(X[te])[:, 1]
        out[tag] = (roc_auc_score(y[te], sc), _recall_at_fpr(y[te], sc))
    return out


def main():
    rows = []
    for name, fname in DATASETS:
        sent, active, contact = load(fname)
        print(f"=== {name}: {len(sent)} nadawcow ===", flush=True)
        for off in (0.0, OFF_RATE):
            tauc = []
            acc = {}
            for seed in SEEDS:
                r = run(sent, active, seed, off)
                for k, (a, rr) in r.items():
                    acc.setdefault(k, {"a": [], "r": []})
                    acc[k]["a"].append(a); acc[k]["r"].append(rr)
                tauc.append(r["temporal"][0])
            line = " | ".join(f"{k} auc={np.mean(v['a']):.3f} r@1%={np.mean(v['r']):.3f}"
                              for k, v in acc.items())
            print(f"  off-hours={off:.0%}: {line}", flush=True)
            for k, v in acc.items():
                rows.append([name, off, k, round(float(np.mean(v["a"])), 4), round(float(np.mean(v["r"])), 4)])
        # istotnosc: temporal (leak-aware) vs losowy 0.5 (n1)
        try:
            stat, p = wilcoxon([a - 0.5 for a in tauc])
            print(f"  Wilcoxon temporal@{OFF_RATE:.0%} vs 0.5: p={p:.4f} (AUC={np.mean(tauc):.3f})")
        except Exception as e:
            print(f"  Wilcoxon: {e}")
    out = RESULTS / "exp_temporal_realgraphs.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["dataset", "off_hours", "model", "auc", "recall_fpr1"])
        w.writerows(rows)
    print(f"\n[temporal-real] -> {out}")


if __name__ == "__main__":
    main()
