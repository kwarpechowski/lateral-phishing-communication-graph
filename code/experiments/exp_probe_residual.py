# -*- coding: utf-8 -*-
"""R4-A2 (analiza resztki sondy) + R4-A3 (zmierzony mechanizm inwersji blast).

Czesc I (resztka sondy, R4-A2):
  Na Enronie w reżimie stealthy/emp sonda czysto czasowa zostaje przy ~0.63 AUC.
  - AUC kazdej pojedynczej cechy sondy (last_gap, mean_gap, gap_std, hist_len, inv_len)
    -> ktora cecha niesie reszte.
  - Temporal GNN z dodanymi cechami sondy na wejsciu (ef + probe features):
    ile AUC zostaje naddetektora bez nich (rozdiel = czesc czytana przez TGN).

Czesc II (inwersja blast, R4-A3):
  Na Enronie i obu SNAP: rozklady cech burst/freshness/propagation atak vs tlo
  (reżim blast), wspolczynnik klasteryzacji grafu, odsetek ofiar hop-1, ktorych
  sasiedzi zostali trafieni wczesniej (wczesniejsza fala vs ten sam kubelek).

Wyniki: results/exp_probe_residual.csv, results/exp_blast_inversion.csv
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

sys.argv = [sys.argv[0]]

import torch  # noqa: E402

from exp_injection_probe import _probe_features, _victim_split  # noqa: E402
from exp_cascade import _index_traffic, WINDOW  # noqa: E402
from exp_cascade_enron import (  # noqa: E402
    _events as enron_events,
    _features as enron_features,
    _compa_features,
    _per_bucket,
    _node_features,
    _train_temporal,
    parse_enron,
    N_BUCKETS,
    SEEDS as EN_SEEDS,
)
from exp_cascade_realgraphs import (  # noqa: E402
    DATASETS,
    _events as snap_events,
    _features as snap_features,
    load_graph,
    T_BUCKETS,
)
from collections import defaultdict as dd


def _probe_single(events, t_buckets, seed):
    """AUC kazdej pojedynczej cechy sondy (wektory kolumnowe)."""
    X = _probe_features(events, t_buckets)
    y = np.array([e[3] for e in events], dtype=np.float32)
    te, tr = _victim_split(events, seed)
    out = {}
    names = ["last_gap", "mean_gap", "gap_std", "hist_len", "inv_len"]
    for j, nm in enumerate(names):
        x = X[:, j]
        if x[tr].std() == 0:
            out[nm] = float("nan")
            continue
        c = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=100, verbose=-1).fit(
            x.reshape(-1, 1)[tr], y[tr]
        )
        s = c.predict_proba(x.reshape(-1, 1)[te])[:, 1]
        out[nm] = round(float(roc_auc_score(y[te], s)), 4)
    return out


def _tgn_with_probe(events, nbr, active, seed, with_probe=True):
    """Temporal GNN na Enronie; with_probe=True dokleja cechy sondy do ef."""
    recv_b, sent_b = _index_traffic(events)
    X1, Xc = enron_features(events, nbr, active, recv_b, sent_b, seed)
    Xp = _probe_features(events, N_BUCKETS)
    y = np.array([e[3] for e in events], dtype=np.float32)
    vics = sorted({e[1] for e in events})
    rng = np.random.default_rng(seed)
    te_v = set(np.array(vics)[rng.permutation(len(vics))[: max(1, len(vics) // 3)]])
    te = np.array([e[1] in te_v for e in events])
    tr = ~te
    nodes = sorted({e[0] for e in events} | {e[1] for e in events})
    idx = {t: i for i, t in enumerate(nodes)}
    n = len(nodes)
    Xn = _node_features(events, nbr, idx, n, recv_b, sent_b, tr=tr)
    ef = torch.tensor(np.hstack([X1, Xp]) if with_probe else X1, dtype=torch.float32)
    s_idx = torch.tensor([idx[e[0]] for e in events])
    v_idx = torch.tensor([idx[e[1]] for e in events])
    yt = torch.tensor(y)
    tr_t = torch.tensor(tr)
    te_t = torch.tensor(te)
    pb = _per_bucket(events, idx)
    auc, r1 = _train_temporal(Xn, pb, n, ef, s_idx, v_idx, yt, tr_t, te_t, y, te, seed)
    return auc, r1


def part_probe_residual():
    print("=== [probe-residual] Enron stealthy/emp: resztka sondy ===", flush=True)
    nbr, sent, active = parse_enron()
    rows = []
    for seed in EN_SEEDS:
        events = enron_events(nbr, sent, seed, stealth_k=2, timing="emp")
        singles = _probe_single(events, N_BUCKETS, seed)
        a0, r0 = _tgn_with_probe(events, nbr, active, seed, with_probe=False)
        a1, r1 = _tgn_with_probe(events, nbr, active, seed, with_probe=True)
        rows.append(
            [seed]
            + [
                singles.get(k, float("nan"))
                for k in ("last_gap", "mean_gap", "gap_std", "hist_len", "inv_len")
            ]
            + [round(a0, 4), round(r0, 4), round(a1, 4), round(r1, 4)]
        )
        print(
            f"  seed={seed}: singles={singles} tgn={a0:.3f} tgn+probe={a1:.3f}",
            flush=True,
        )
    out = RESULTS / "exp_probe_residual.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "seed",
                "auc_last_gap",
                "auc_mean_gap",
                "auc_gap_std",
                "auc_hist_len",
                "auc_inv_len",
                "tgn_auc",
                "tgn_r1",
                "tgn_probe_auc",
                "tgn_probe_r1",
            ]
        )
        w.writerows(rows)
    print(f"[probe-residual] -> {out}")


# ============ R4-A3: inwersja blast ============


def _feature_stats(events, nbr, seed, features_fn, t_buckets):
    """Srednie cech burst/freshness/propagation per klasa (wektor cech ctx)."""
    recv_b, sent_b = _index_traffic(events)
    r = features_fn(events, nbr, recv_b, sent_b, seed)
    Xc = r[1]  # enron: (X1, Xc); realgraphs: (X1, Xc, y)
    y = np.array([e[3] for e in events], dtype=np.float32)
    # Xc kolumny: [s_in_contacts, deg_v, deg_s, burst, freshness, propagation]
    cols = {"burst": 3, "freshness": 4, "propagation": 5}
    out = {}
    for nm, j in cols.items():
        out[f"{nm}_attack"] = float(Xc[y == 1, j].mean())
        out[f"{nm}_benign"] = float(Xc[y == 0, j].mean())
    return out


def _clustering(nbr):
    """Wspolczynnik klasteryzacji (przyblizenie: srednia po wezlach lokalnych)."""
    tot, cnt = 0.0, 0
    for u in list(nbr)[:2000]:
        ns = nbr.get(u, set())
        k = len(ns)
        if k < 2:
            continue
        links = 0
        ns = list(ns)
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                if ns[j] in nbr.get(ns[i], set()):
                    links += 1
        tot += 2 * links / (k * (k - 1))
        cnt += 1
    return tot / max(1, cnt)


def _hop1_wave(events, t_buckets):
    """Odsetek zdarzen ataku hop-1 (cel = kontakt nosiciela z t0), ktorych INNI
    sasiedzi ofiary zostali trafieni SCISLE W ранniej (fala widoczna w oknie strict-past)
    vs w TYM SAMYM kubełku."""
    by_bucket = dd(list)
    for s, v, b, lab in events:
        if lab == 1:
            by_bucket[b].append((s, v))
    recv_hit = dd(set)  # bucket -> victims
    for b in by_bucket:
        for s, v in by_bucket[b]:
            recv_hit[b].add(v)
    # prefix per bucket w kolejnosci listy (regula remisow = kolejnosc w events)
    same_bucket = earlier = total = 0
    seen = set()
    for s, v, b, lab in events:
        if lab != 1:
            continue
        # inni nadawcy trafili v wczesniej? (b-w .. b-1)
        prev = set()
        for bb in range(max(0, b - WINDOW), b):
            prev |= recv_hit.get(bb, set())
        if v in prev:
            earlier += 1
        # inni nadawcy trafili v w tym samym kubełku (prefix = pozycja w liscie)?
        total += 1
    # odsetek ofiar-multi-hit: ofiary ataku trafione wiecej niz raz (caly strumien)
    vic_count = dd(int)
    for s, v, b, lab in events:
        if lab == 1:
            vic_count[v] += 1
    multi = sum(1 for c in vic_count.values() if c > 1) / max(1, len(vic_count))
    return {
        "attack_events_hit_earlier_frac": round(earlier / max(1, total), 4),
        "attack_victims_multi_hit_frac": round(multi, 4),
    }


def part_blast_inversion():
    print("=== [blast-inversion] cechy + klasteryzacja + fala hop-1 ===", flush=True)
    rows = []
    # Enron (blast)
    nbr, sent, active = parse_enron()
    cl = _clustering(nbr)
    for seed in EN_SEEDS[:4]:
        events = enron_events(nbr, sent, seed, stealth_k=None)
        fs = _feature_stats(events, nbr, seed, enron_features, N_BUCKETS)
        wv = _hop1_wave(events, N_BUCKETS)
        rows.append(
            ["Enron", "blast", seed, round(cl, 4)]
            + [
                fs[k]
                for k in (
                    "burst_attack",
                    "burst_benign",
                    "freshness_attack",
                    "freshness_benign",
                    "propagation_attack",
                    "propagation_benign",
                )
            ]
            + [
                wv["attack_events_hit_earlier_frac"],
                wv["attack_victims_multi_hit_frac"],
            ]
        )
        print(f"  Enron blast seed={seed}: {fs} {wv}", flush=True)
    # SNAP (blast)
    for name, fname in DATASETS:
        nbr = load_graph(fname)
        cl = _clustering(nbr)
        for seed in range(4):
            events = snap_events(nbr, seed, stealth_k=None)
            fs = _feature_stats(events, nbr, seed, snap_features, T_BUCKETS)
            wv = _hop1_wave(events, T_BUCKETS)
            rows.append(
                [name, "blast", seed, round(cl, 4)]
                + [
                    fs[k]
                    for k in (
                        "burst_attack",
                        "burst_benign",
                        "freshness_attack",
                        "freshness_benign",
                        "propagation_attack",
                        "propagation_benign",
                    )
                ]
                + [
                    wv["attack_events_hit_earlier_frac"],
                    wv["attack_victims_multi_hit_frac"],
                ]
            )
            print(f"  {name} blast seed={seed}: {fs} {wv}", flush=True)
    out = RESULTS / "exp_blast_inversion.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "graph",
                "regime",
                "seed",
                "clustering",
                "burst_attack",
                "burst_benign",
                "freshness_attack",
                "freshness_benign",
                "propagation_attack",
                "propagation_benign",
                "hit_earlier_frac",
                "multi_hit_frac",
            ]
        )
        w.writerows(rows)
    print(f"[blast-inversion] -> {out}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    if len(sys.argv) > 1 and sys.argv[1] == "inversion":
        part_blast_inversion()
    else:
        part_probe_residual()
        part_blast_inversion()
