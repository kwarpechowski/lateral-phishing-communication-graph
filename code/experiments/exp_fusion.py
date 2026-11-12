"""#GP-EXP-3 (H8 climax): does FUSING all layers reach a deployable operating point?

#GP-EXP-1 (OSINT layers) fixed the new-sender FP; #GP-EXP-2 (temporal) fixed the
compromised-account FN — but each ALONE left Recall@FPR=1% at 0 (its own noise floods a
1% threshold). The operational question: does the CONJUNCTION of layer-inconsistencies
(cross-layer consistency, H8c) push Recall@FPR=1% above 0, where no single layer can?

Unified event model (one population, both benign kinds + both attack kinds):
  benign known     : contact sender, in-rhythm                       label 0
  benign new_osint : off-contact but on-OSINT sender, in-rhythm      label 0  (FP risk: contact)
  benign new_cold  : off everything sender, in-rhythm                label 0  (FP risk: all static)
  attack bec_off   : off everything sender, off-rhythm               label 1
  attack compromised: contact sender, off-rhythm                     label 1  (FN risk: static)
with two-sided noise (some benign off-hours; some attackers mimic the rhythm).

Feature sets (LightGBM, split by victim, 5 seeds), headline = Recall@FPR=1%:
  contact_only : [in contact graph]                                  (the seed baseline)
  static       : contact + OSINT layers + coverage                   (#GP-EXP-1 multiplex)
  full         : static + temporal-consistency                       (all layers fused)
  full_shuf    : full but all layers/rhythm re-wired to random twins (causal control)

Output: results/exp_fusion.csv  (overall + per-attack-type Recall@FPR1%).
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
from graph.build_temporal_overlay import (active_buckets, in_bucket,  # noqa: E402
                                          is_consistent, off_bucket)

SEEDS = list(range(20))   # >=20 ziaren dla istotnosci (Wilcoxon n=5 ma podloge p=0.0625)
EVENTS_PER_VICTIM = 24
BENIGN_MIX = (0.40, 0.30, 0.30)      # known / new_osint / new_cold
BENIGN_OFFHOURS = 0.15
ATTACKER_MIMICS = 0.25


def _h(*p: str) -> int:
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def _recall_at_fpr(y, s, t=0.01) -> float:
    y = np.asarray(y); s = np.asarray(s)
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


# Warstwy OSINT z REALNYCH pol blizniaka (c3/c4/c5), nie syntetyczne spolecznosci.
# Pomijamy zbyt rzadkie (osint_oss=3, osint_publication=8 krawedzi) oraz zdegenerowane
# (osint_techstack, likely_senders ~ near-complete: generyczne wartosci lacza wszystkich).
REAL_OSINT = ["osint_conference", "osint_certification", "osint_skill", "osint_routine",
              "osint_event", "osint_pretext", "osint_platform", "interest_sim"]


def run(seed):
    base = build_layers()
    layers = {"contact": base["contact"], **{L: base[L] for L in REAL_OSINT}}
    names = list(layers)
    adj = _adj(layers)
    contact_nbr = adj["contact"]
    osint_union = {}
    for L in names[1:]:
        for t, nb in adj[L].items():
            osint_union.setdefault(t, set()).update(nb)
    twins = sorted(set(contact_nbr) | set(osint_union) | {r["twin_id"] for r in _self_rows()})
    wrong = {twins[i]: twins[(i + 7) % len(twins)] for i in range(len(twins))}

    rows = []   # (sender, victim, label, atk_type, bucket)
    for v in twins:
        c = sorted(contact_nbr.get(v, set()))
        o_only = sorted(osint_union.get(v, set()) - contact_nbr.get(v, set()) - {v})
        off = [t for t in twins if t != v and t not in contact_nbr.get(v, set())
               and t not in osint_union.get(v, set())]
        if not c or not off:
            continue
        for k in range(EVENTS_PER_VICTIM):
            h = _h(str(seed), v, str(k)); salt = f"{v}:{k}"
            if h % 2 == 0:                                          # benign
                r = (h // 3) % 100
                if r < BENIGN_MIX[0] * 100:
                    s = c[h % len(c)]
                elif r < (BENIGN_MIX[0] + BENIGN_MIX[1]) * 100 and o_only:
                    s = o_only[h % len(o_only)]
                else:
                    s = off[h % len(off)]
                b = off_bucket(s, salt, seed) if (h // 7) % 100 < int(BENIGN_OFFHOURS * 100) else in_bucket(s, salt, seed)
                rows.append((s, v, 0, "benign", b))
            else:                                                  # attack
                mimic = (h // 5) % 100 < int(ATTACKER_MIMICS * 100)
                if (h // 11) % 2 == 0:                              # BEC off-graph
                    s = off[h % len(off)]
                    b = in_bucket(s, salt, seed) if mimic else off_bucket(s, salt, seed)
                    rows.append((s, v, 1, "bec_off", b))
                else:                                              # compromised contact
                    s = c[h % len(c)]
                    b = in_bucket(s, salt, seed) if mimic else off_bucket(s, salt, seed)
                    rows.append((s, v, 1, "compromised", b))

    y = np.array([r[2] for r in rows]); atk = np.array([r[3] for r in rows]); vic = [r[1] for r in rows]
    order = sorted(set(vic))
    te_v = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order)//3)]])
    te = np.array([x in te_v for x in vic]); tr = ~te

    def feats(victim_map, rhythm_map):
        Xc, Xs, Xf = [], [], []
        for s, v, _l, _a, b in rows:
            vv = victim_map(v)
            per = [1.0 if s in adj[L].get(vv, set()) else 0.0 for L in names]
            cov = sum(per[1:])
            tcon = 1.0 if is_consistent(rhythm_map(s), b, seed) else 0.0
            Xc.append([per[0]]); Xs.append(per + [cov]); Xf.append(per + [cov, tcon])
        return (np.array(Xc, np.float32), np.array(Xs, np.float32), np.array(Xf, np.float32))

    Xc, Xs, Xf = feats(lambda v: v, lambda s: s)
    _, _, Xf_sh = feats(lambda v: wrong[v], lambda s: wrong[s])

    def fit(X):
        clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=250, verbose=-1).fit(X[tr], y[tr])
        return clf.predict_proba(X[te])[:, 1]
    yte = y[te]; atk_te = atk[te]
    out = {}
    for tag, X in (("contact", Xc), ("static", Xs), ("full", Xf), ("full_shuf", Xf_sh)):
        s = fit(X)
        out[f"{tag}_auc"] = roc_auc_score(yte, s)
        out[f"{tag}_r1"] = _recall_at_fpr(yte, s)
        if tag == "full":   # per-attack-type recall@1% (which attacks the fusion catches)
            thr = np.quantile(s[yte == 0], 0.99)
            for a in ("bec_off", "compromised"):
                m = (yte == 1) & (atk_te == a)
                out[f"full_r1_{a}"] = float((s[m] >= thr).mean()) if m.any() else 0.0
    return out


def _self_rows():
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    from cascadebench.stats import mean_ci          # bootstrap-CI95 (percentylowy)
    acc = {}
    for seed in SEEDS:
        for k, v in run(seed).items():
            acc.setdefault(k, []).append(v)
    agg = {k: float(np.mean(v)) for k, v in acc.items()}
    ci = {k: mean_ci(v) for k, v in acc.items()}     # (mean, lo, hi)
    out = RESULTS / "exp_fusion.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["metric", "mean", "ci_lo", "ci_hi", "n_seeds"])
        for k, v in acc.items():
            m, lo, hi = ci[k]
            w.writerow([k, round(m, 4), round(lo, 4), round(hi, 4), len(v)])
    sd = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in acc.items()}
    print("[exp-fusion] H8 climax — does fusing layers reach Recall@FPR=1% > 0?")
    print(f"  {'model':12s} {'AUC (mean+-sd)':>20s} {'Recall@FPR1% (mean+-sd)':>26s}")
    for tag in ("contact", "static", "full", "full_shuf"):
        print(f"  {tag:12s} {agg[tag+'_auc']:.3f}+-{sd[tag+'_auc']:.3f}      "
              f"{agg[tag+'_r1']:.3f}+-{sd[tag+'_r1']:.3f}")
    print(f"\n  full Recall@FPR1% by attack: bec_off={agg['full_r1_bec_off']:.3f} "
          f"compromised={agg['full_r1_compromised']:.3f}")
    print(f"[exp-fusion] wrote {out}")


if __name__ == "__main__":
    main()
