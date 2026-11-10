"""#GP-EXP-2 (H8b): can a TEMPORAL layer recover the compromised-account blind spot?

A compromised in-graph account is the seed signal's worst case: the taken-over sender IS
a known contact, so contact + every static OSINT layer say "legit" (AUC ~0.51, the
provenance blind spot). The hypothesis (H8b): the malicious email arrives OFF the
sender's usual rhythm, so a temporal-consistency feature recovers it.

Events (all senders are KNOWN contacts of the victim — that is the point):
  benign      : known sender, email IN the sender's active buckets (mostly), label 0.
  compromised : known sender, email OFF the sender's buckets (mostly), label 1.
With two-sided noise (some legit off-hours mail; some attackers mimic the rhythm) so the
temporal signal is strong but not a perfect-separability artifact.

Feature sets (LightGBM, split by victim, 5 seeds):
  static       : contact + per-OSINT-layer membership + coverage  (blind to compromise)
  +temporal    : static + [email fits sender rhythm?] + [co-active with victim?]
  +temporal_shuf: temporal feature computed against a WRONG sender's rhythm (control)

Output: results/exp_compromised.csv
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

SEEDS = list(range(20))   # >=20 ziaren dla istotnosci
EVENTS_PER_VICTIM = 16
BENIGN_OFFHOURS = 0.15     # legit mail occasionally off-rhythm (FP risk for temporal)
ATTACKER_MIMICS = 0.25     # smart attacker occasionally hits the rhythm (FN risk)


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


REAL_OSINT = ["osint_conference", "osint_certification", "osint_skill", "osint_routine",
              "osint_event", "osint_pretext", "osint_platform", "interest_sim"]


def run(seed):
    base = build_layers()
    layers = {"contact": base["contact"], **{L: base[L] for L in REAL_OSINT}}
    names = list(layers)
    adj = _adj(layers)
    contact_nbr = adj["contact"]
    twins = sorted(contact_nbr)
    wrong = {twins[i]: twins[(i + 7) % len(twins)] for i in range(len(twins))}

    rows = []   # (sender, victim, label, bucket)
    for v in twins:
        c = sorted(contact_nbr.get(v, set()))
        if not c:
            continue
        for k in range(EVENTS_PER_VICTIM):
            h = _h(str(seed), v, str(k))
            s = c[h % len(c)]                                   # sender is ALWAYS a known contact
            salt = f"{v}:{k}"
            if h % 2 == 0:                                      # benign
                if (h // 3) % 100 < int(BENIGN_OFFHOURS * 100):
                    b = off_bucket(s, salt, seed)              # legit but off-hours (noise)
                else:
                    b = in_bucket(s, salt, seed)
                rows.append((s, v, 0, b))
            else:                                              # compromised attack
                if (h // 5) % 100 < int(ATTACKER_MIMICS * 100):
                    b = in_bucket(s, salt, seed)               # attacker mimics rhythm (noise)
                else:
                    b = off_bucket(s, salt, seed)
                rows.append((s, v, 1, b))

    y = np.array([r[2] for r in rows]); vic = [r[1] for r in rows]
    order = sorted(set(vic))
    te_v = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order)//3)]])
    te = np.array([x in te_v for x in vic]); tr = ~te

    def static_feat(s, v):
        per = [1.0 if s in adj[L].get(v, set()) else 0.0 for L in names]
        return per + [sum(per[1:])]

    Xstat = np.array([static_feat(s, v) for s, v, _l, _b in rows], dtype=np.float32)
    tcon = np.array([[1.0 if is_consistent(s, b, seed) else 0.0] for s, v, _l, b in rows], dtype=np.float32)
    tshuf = np.array([[1.0 if is_consistent(wrong[s], b, seed) else 0.0] for s, v, _l, b in rows], dtype=np.float32)
    Xtemp = np.hstack([Xstat, tcon])
    Xtshf = np.hstack([Xstat, tshuf])

    def fit(X):
        clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(X[tr], y[tr])
        return clf.predict_proba(X[te])[:, 1]
    yte = y[te]
    out = {}
    for tag, X in (("static", Xstat), ("temporal", Xtemp), ("temporal_shuf", Xtshf)):
        s = fit(X)
        out[f"{tag}_auc"] = roc_auc_score(yte, s)
        out[f"{tag}_r1"] = _recall_at_fpr(yte, s)
    return out


def main():
    acc = {}
    for seed in SEEDS:
        r = run(seed)
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    agg = {k: float(np.mean(v)) for k, v in acc.items()}
    out = RESULTS / "exp_compromised.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["metric", "mean", "std", "n_seeds"])
        for k, v in acc.items():
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            w.writerow([k, round(float(np.mean(v)), 4), round(sd, 4), len(v)])
    print("[exp-compromised] H8b: detecting compromised in-graph accounts")
    print(f"  {'model':16s} {'AUC':>7s} {'Recall@FPR1%':>13s}")
    for tag in ("static", "temporal", "temporal_shuf"):
        print(f"  {tag:16s} {agg[tag+'_auc']:7.3f} {agg[tag+'_r1']:13.3f}")
    print(f"\n  -> static is the seed blind spot (~0.5); temporal should recover it; "
          f"shuffled must NOT.\n[exp-compromised] wrote {out}")


if __name__ == "__main__":
    main()
