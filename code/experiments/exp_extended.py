"""Rozszerzone eksperymenty multipleksu (#GP-EXP-4..6): ablacja + sweepy.

Dok{\l}ada do EXP-1/2/3 eksperymenty z hypotheses.md, kt{\'o}re jeszcze nie by{\l}y pokryte:
  * ABLACJA warstw (H8d): kt{\'o}re warstwy realnie nios{\k{a}} sygna{\l} (cumulative-add +
    leave-one-out). Oczekiwany negatyw dla s{\l}abej warstwy zainteresowa{\'n}/OSINT.
  * SWEEP p_cross (overlap cross-org): jak overlap mi{\k{e}}dzy-organizacyjny warstw OSINT
    wp{\l}ywa na AUC/Recall (mosty nowych nadawc{\'o}w, H9a).
  * SWEEP fabrication (H9d): adwersaryjny crossover — fa{\l}szywe kraw{\k{e}}dzie OSINT eroduj{\k{a}}
    przewag{\k{e}} multipleksu do poziomu kontroli przetasowanej.

Wsp{\'o}lny model zdarze{\'n} z EXP-3 (benign known/new_osint/new_cold; atak bec_off/compromised),
LightGBM, podzia{\l} po ofiarach, 5 ziaren, metryka Recall@FPR=1\%. Bez LLM/GNN.

Wyj{\'s}cie: results/exp_ablation.csv, results/exp_sweep_pcross.csv, results/exp_sweep_fab.csv.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from graph.build_layers import build_layers                          # noqa: E402
from graph.build_osint_overlay import build_overlay                  # noqa: E402
from graph.build_temporal_overlay import in_bucket, is_consistent, off_bucket  # noqa: E402

SEEDS = list(range(5))
EVENTS_PER_VICTIM = 24
BENIGN_MIX = (0.40, 0.30, 0.30)
BENIGN_OFFHOURS, ATTACKER_MIMICS = 0.15, 0.25


def _h(*p):
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def _recall_at_fpr(y, s, t=0.01):
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(y, s)
    m = fpr <= t
    return float(tpr[m].max()) if m.any() else 0.0


def _adj(layers):
    out = {}
    for name, edges in layers.items():
        nbr = {}
        for e in edges:
            nbr.setdefault(e[0], set()).add(e[1])
            nbr.setdefault(e[1], set()).add(e[0])
        out[name] = nbr
    return out


# Realne warstwy OSINT z pol blizniaka (c3-c8) — do ablacji; sweepy p_cross/fabrykacji
# zostaja na syntetycznym overlay (pokretla nie istnieja dla realnych warstw).
REAL_OSINT = ["osint_conference", "osint_certification", "osint_skill", "osint_routine",
              "osint_event", "osint_pretext", "osint_platform", "interest_sim"]


def _prepare(seed, p_cross, attack_fab=0.0, real=False):
    """attack_fab: prob. that a BEC sender PLANTS a fake OSINT edge to the victim (H9d)."""
    base = build_layers()
    if real:
        layers = {"contact": base["contact"], **{L: base[L] for L in REAL_OSINT}}
    else:
        ov, _ = build_overlay(seed=seed, p_cross=p_cross, fabrication_rate=0.0)
        layers = {"contact": base["contact"], **ov}
    names = list(layers)
    adj = _adj(layers)
    contact_nbr = adj["contact"]
    osint_union = {}
    for L in names[1:]:
        for t, nb in adj[L].items():
            osint_union.setdefault(t, set()).update(nb)
    twins = sorted(set(contact_nbr) | set(osint_union) |
                   {r["twin_id"] for r in _self_rows()})
    osint_layers = names[1:]
    rows = []   # (s, v, label, type, bucket, fab_layer)
    for v in twins:
        c = sorted(contact_nbr.get(v, set()))
        o_only = sorted(osint_union.get(v, set()) - contact_nbr.get(v, set()) - {v})
        off = [t for t in twins if t != v and t not in contact_nbr.get(v, set())
               and t not in osint_union.get(v, set())]
        if not c or not off:
            continue
        for k in range(EVENTS_PER_VICTIM):
            h = _h(str(seed), v, str(k)); salt = f"{v}:{k}"
            if h % 2 == 0:
                r = (h // 3) % 100
                if r < BENIGN_MIX[0] * 100:
                    s, st = c[h % len(c)], "known"
                elif r < (BENIGN_MIX[0] + BENIGN_MIX[1]) * 100 and o_only:
                    s, st = o_only[h % len(o_only)], "new_osint"
                else:
                    s, st = off[h % len(off)], "new_cold"
                b = off_bucket(s, salt, seed) if (h // 7) % 100 < int(BENIGN_OFFHOURS * 100) else in_bucket(s, salt, seed)
                rows.append((s, v, 0, st, b, None))
            else:
                mimic = (h // 5) % 100 < int(ATTACKER_MIMICS * 100)
                if (h // 11) % 2 == 0:
                    s = off[h % len(off)]
                    b = in_bucket(s, salt, seed) if mimic else off_bucket(s, salt, seed)
                    fab = osint_layers[h % len(osint_layers)] if (h // 13) % 100 < int(attack_fab * 100) else None
                    rows.append((s, v, 1, "bec_off", b, fab))
                else:
                    s = c[h % len(c)]
                    b = in_bucket(s, salt, seed) if mimic else off_bucket(s, salt, seed)
                    rows.append((s, v, 1, "compromised", b, None))
    return adj, names, rows, twins


def _self_rows():
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _eval(adj, rows, twins, seed, layer_subset, use_temporal, shuffle=False):
    wrong = {twins[i]: twins[(i + 7) % len(twins)] for i in range(len(twins))}
    y = np.array([r[2] for r in rows]); vic = [r[1] for r in rows]
    order = sorted(set(vic))
    te_v = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order)//3)]])
    te = np.array([x in te_v for x in vic]); tr = ~te
    X = []
    for s, v, _l, _a, b, fab in rows:
        vv = wrong[v] if shuffle else v
        ss = wrong[s] if shuffle else s
        per = [1.0 if (s in adj[L].get(vv, set()) or L == fab) else 0.0 for L in layer_subset]
        feat = per + [sum(per)]
        if use_temporal:
            feat.append(1.0 if is_consistent(ss, b, seed) else 0.0)
        X.append(feat)
    X = np.array(X, np.float32)
    clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(X[tr], y[tr])
    sc = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y[te], sc), _recall_at_fpr(y[te], sc)


def ablation():
    """Cumulative-add and leave-one-out over layers (H8d)."""
    osint = REAL_OSINT
    pl = {"osint_conference": "+konferencje", "osint_certification": "+certyfikaty",
          "osint_skill": "+umiejetnosci", "osint_routine": "+rutyna", "osint_event": "+wydarzenia",
          "osint_pretext": "+pretexty", "osint_platform": "+platformy", "interest_sim": "+zainteresowania"}
    configs = {"kontakty": (["contact"], False)}
    cum = ["contact"]
    for L in osint:                                    # cumulative-add po REALNYCH warstwach
        cum = cum + [L]
        configs[pl[L]] = (list(cum), False)            # ostatnia (+zainteresowania) = wszystkie OSINT
    configs["+czasowa (pelny)"] = (["contact"] + osint, True)
    acc = {k: {"auc": [], "r1": []} for k in configs}
    for seed in SEEDS:
        adj, names, rows, twins = _prepare(seed, 0.5, 0.0, real=True)
        for k, (subset, temp) in configs.items():
            a, r = _eval(adj, rows, twins, seed, subset, temp)
            acc[k]["auc"].append(a); acc[k]["r1"].append(r)
    out = RESULTS / "exp_ablation.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["config", "auc", "recall_fpr1"])
        for k in configs:
            w.writerow([k, round(np.mean(acc[k]["auc"]), 4), round(np.mean(acc[k]["r1"]), 4)])
    print(f"[ablation] -> {out}")
    for k in configs:
        print(f"  {k:22s} AUC={np.mean(acc[k]['auc']):.3f}  R@1%={np.mean(acc[k]['r1']):.3f}")


def sweep_pcross():
    grid = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    out = RESULTS / "exp_sweep_pcross.csv"
    osint = ["osint_mailing_list", "osint_oss_project", "osint_affiliation"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["p_cross", "static_auc", "static_recall_fpr1"])
        for pc in grid:
            aus, r1s = [], []
            for seed in SEEDS:
                adj, names, rows, twins = _prepare(seed, pc, 0.0)
                a, r = _eval(adj, rows, twins, seed, ["contact"] + osint, False)
                aus.append(a); r1s.append(r)
            w.writerow([pc, round(np.mean(aus), 4), round(np.mean(r1s), 4)])
            print(f"[pcross] p_cross={pc}: static AUC={np.mean(aus):.3f}")
    print(f"[pcross] -> {out}")


def sweep_fab():
    grid = [0.0, 0.1, 0.2, 0.3, 0.5]
    out = RESULTS / "exp_sweep_fab.csv"
    osint = ["osint_mailing_list", "osint_oss_project", "osint_affiliation"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["fabrication", "full_auc", "shuffled_auc"])
        for fab in grid:
            full, shuf = [], []
            for seed in SEEDS:
                adj, names, rows, twins = _prepare(seed, 0.5, fab)
                a, _ = _eval(adj, rows, twins, seed, ["contact"] + osint, True)
                a_s, _ = _eval(adj, rows, twins, seed, ["contact"] + osint, True, shuffle=True)
                full.append(a); shuf.append(a_s)
            w.writerow([fab, round(np.mean(full), 4), round(np.mean(shuf), 4)])
            print(f"[fab] fabrication={fab}: full AUC={np.mean(full):.3f} shuf={np.mean(shuf):.3f}")
    print(f"[fab] -> {out}")


def main():
    print("=== ablacja warstw (H8d) ==="); ablation()
    print("\n=== sweep p_cross (H9a) ==="); sweep_pcross()
    print("\n=== sweep fabrication (H9d) ==="); sweep_fab()


if __name__ == "__main__":
    main()
