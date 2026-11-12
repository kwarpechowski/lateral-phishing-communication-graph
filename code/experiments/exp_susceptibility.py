"""#GP-EXP-SUSC (paper C&S): organizational susceptibility, MEASURED by our method.

The C&S synthesis paper defines susceptibility of a node v as the worst-case
residual miss of the fused multiplex detector over the attacks that can reach v:

    Susc(v) = max_{a in A(v)} ( 1 - Pr[detect(a)] )           (eq:susc)

We do NOT invent these numbers: we reuse the EXACT fused detector and unified event
model of #GP-EXP-3 (exp_fusion.py) and simply break the operational Recall@FPR=1%
down by (i) the victim's seniority and (ii) the attack channel / relation type.
Susceptibility is then 1 - Recall@FPR1% (the share of attacks that slip through at
the deployable 1% operating point).

Seniority comes from results/twin_self.csv (role -> seniority):
  Managing Director -> executive | Senior Manager -> senior
  Specialist -> mid              | Analyst -> junior

Channels (relation of the attacker to the victim):
  bec_off     : external impersonation, sender off EVERY layer (cold)
  compromised : lateral, sender is a known contact, off-rhythm

Output: results/exp_susceptibility.csv
  rows: group_kind (seniority|channel|overall), group, recall_fpr1, susc, n_attacks, n_seeds
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from graph.build_layers import build_layers                          # noqa: E402
from graph.build_temporal_overlay import (in_bucket, is_consistent,   # noqa: E402
                                          off_bucket)

# --- mirror exp_fusion.py exactly so numbers stay consistent with P1 ---------
SEEDS = list(range(20))
EVENTS_PER_VICTIM = 24
BENIGN_MIX = (0.40, 0.30, 0.30)      # known / new_osint / new_cold
BENIGN_OFFHOURS = 0.15
ATTACKER_MIMICS = 0.25
REAL_OSINT = ["osint_conference", "osint_certification", "osint_skill", "osint_routine",
              "osint_event", "osint_pretext", "osint_platform", "interest_sim"]

_SENIORITY = {"Managing Director": "executive", "Senior Manager": "senior",
              "Specialist": "mid", "Analyst": "junior"}


def _h(*p: str) -> int:
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def _adj(layers):
    out = {}
    for name, edges in layers.items():
        nbr = {}
        for e in edges:
            nbr.setdefault(e[0], set()).add(e[1])
            nbr.setdefault(e[1], set()).add(e[0])
        out[name] = nbr
    return out


def _self_rows():
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _seniority_map():
    return {r["twin_id"]: _SENIORITY.get(r["role"], "mid") for r in _self_rows()}


def run(seed, sen_map):
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
                if (h // 11) % 2 == 0:
                    s = off[h % len(off)]
                    b = in_bucket(s, salt, seed) if mimic else off_bucket(s, salt, seed)
                    rows.append((s, v, 1, "bec_off", b))
                else:
                    s = c[h % len(c)]
                    b = in_bucket(s, salt, seed) if mimic else off_bucket(s, salt, seed)
                    rows.append((s, v, 1, "compromised", b))

    y = np.array([r[2] for r in rows]); atk = np.array([r[3] for r in rows]); vic = [r[1] for r in rows]
    order = sorted(set(vic))
    te_v = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order)//3)]])
    te = np.array([x in te_v for x in vic]); tr = ~te

    # full (fused) feature set only — the deployable detector
    Xf = []
    for s, v, _l, _a, b in rows:
        per = [1.0 if s in adj[L].get(v, set()) else 0.0 for L in names]
        cov = sum(per[1:])
        tcon = 1.0 if is_consistent(s, b, seed) else 0.0
        Xf.append(per + [cov, tcon])
    Xf = np.array(Xf, np.float32)

    clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=250, verbose=-1).fit(Xf[tr], y[tr])
    score = clf.predict_proba(Xf[te])[:, 1]
    yte = y[te]; atk_te = atk[te]
    vic_te = [vic[i] for i in range(len(rows)) if te[i]]
    sen_te = np.array([sen_map.get(v, "mid") for v in vic_te])

    thr = np.quantile(score[yte == 0], 0.99)         # FPR=1% operating point GLOBALNY
    auc = roc_auc_score(yte, score) if yte.sum() and (yte == 0).sum() else float("nan")

    # B4 (recenzja wewnetrzna r2): dodatkowo prog PER GRUPA — kwantyl 99% benign
    # wewnatrz danej grupy (FPR=1% kontrolowane w grupie, nie globalnie).
    def thr_of(mask_benign):
        s = score[mask_benign & (yte == 0)]
        return np.quantile(s, 0.99) if len(s) >= 10 else thr

    res = {"overall": ((float((score[(yte == 1)] >= thr).mean()), int((yte == 1).sum())), auc)}
    masks = {}
    for a in ("bec_off", "compromised"):
        masks[f"channel:{a}"] = atk_te == a
    for sv in ("executive", "senior", "mid", "junior"):
        masks[f"seniority:{sv}"] = sen_te == sv
        for a in ("bec_off", "compromised"):
            masks[f"seniority:{sv}|{a}"] = (sen_te == sv) & (atk_te == a)
    for k, mask in masks.items():
        m = mask & (yte == 1)
        r_g = (float((score[m] >= thr).mean()), int(m.sum())) if m.any() else (float("nan"), 0)
        res[k] = (r_g, auc)
        thr_g = thr_of(mask)
        r_pg = (float((score[m] >= thr_g).mean()), int(m.sum())) if m.any() else (float("nan"), 0)
        res[k + "|per_group_thr"] = (r_pg, auc)
    return res


def main():
    sen_map = _seniority_map()
    acc = {}
    for seed in SEEDS:
        for k, (rec, _auc) in run(seed, sen_map).items():
            r, n = rec
            if not np.isnan(r):
                acc.setdefault(k, {"rec": [], "n": []})
                acc[k]["rec"].append(r); acc[k]["n"].append(n)

    out = RESULTS / "exp_susceptibility.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "recall_fpr1", "susc", "n_attacks_mean", "n_seeds"])
        for k in sorted(acc):
            rec = np.array(acc[k]["rec"]); n = np.array(acc[k]["n"])
            mr = float(rec.mean())
            w.writerow([k, round(mr, 4), round(1 - mr, 4), round(float(n.mean()), 1), len(rec)])

    print("[exp-susceptibility] Susc(group) = 1 - Recall@FPR1% (fused detector, 20 seeds)\n")
    print(f"  {'group':28s} {'Recall@1%':>10s} {'Susc':>8s}")
    for k in sorted(acc):
        rec = np.array(acc[k]["rec"]); mr = float(rec.mean())
        print(f"  {k:28s} {mr:>10.3f} {1-mr:>8.3f}")
    print(f"\n[exp-susceptibility] wrote {out}")


if __name__ == "__main__":
    main()
