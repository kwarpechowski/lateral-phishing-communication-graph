"""#GP-EXP-1 (feature-level, no GNN): does MULTIPLEX fusion beat the single contact graph?

A fast, dependency-light pre-test of H8a/H8c + H9a/H9c/H9d using LightGBM on graph
features (no PyTorch Geometric, no text). Reproduces the seed setup (classify a
(sender->victim) email event as benign vs BEC by sender provenance) but replaces the
single binary "sender in contact graph" feature with the full MULTIPLEX:

  * benign known-sender : S is a contact of V (in the contact layer).
  * benign NEW-sender   : S is NOT a contact of V but IS connected via an OSINT layer
                          (shared mailing list / OSS / affiliation) — a legit first
                          contact. This is the false-positive source that collapsed the
                          seed signal: contact-only cannot tell it from an attack.
  * BEC attack          : spoofed S connected to V in NO genuine layer. With a fabrication
                          rate, the attacker plants a fake OSINT edge (H9d).

Feature sets compared (LightGBM, split by victim, Recall@FPR=1% is the headline metric):
  contact_only : [in contact graph]                         (the seed baseline)
  multiplex    : per-layer membership + cross-layer coverage (#layers, sum-weight, has-OSINT)
  shuffled     : multiplex but layers re-wired to random twins (causal control — must NOT help)

Sweeps p_cross (cross-org OSINT bridging) and fabrication_rate (H9d). Pure local data
(150-twin multiplex); no LLM, no network.

Output: results/exp_multiplex.csv
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

from graph.build_layers import build_layers                     # noqa: E402
from graph.build_osint_overlay import build_overlay             # noqa: E402

SEEDS = list(range(20))   # >=20 ziaren dla istotnosci
# Realistic two-sided noise (avoid the synthetic perfect-separability artifact):
#   benign senders: known / new-on-OSINT (multiplex can rescue) / new-COLD (no tie -> FP for both)
#   BEC senders:    off-graph spoof / fabricated-OSINT spoof (planted fake footprint, H9d)
P_KNOWN, P_NEW_OSINT, P_NEW_COLD = 0.40, 0.35, 0.25     # benign mix
EVENTS_PER_VICTIM = 14


def _h(*p: str) -> int:
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def _recall_at_fpr(y, s, fpr_target=0.01) -> float:
    y = np.asarray(y); s = np.asarray(s)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(y, s)
    mask = fpr <= fpr_target
    return float(tpr[mask].max()) if mask.any() else 0.0


def _adj(layers: dict) -> dict:
    """layer -> {twin: set(neighbours)} (undirected)."""
    out = {}
    for name, edges in layers.items():
        nbr: dict[str, set] = {}
        for e in edges:
            a, b = e[0], e[1]
            nbr.setdefault(a, set()).add(b)
            nbr.setdefault(b, set()).add(a)
        out[name] = nbr
    return out


def _events(twins, contact_nbr, osint_union, osint_layers, seed, fabrication_rate):
    """Yield (sender, victim, label, stype, fab_layer) with realistic two-sided noise.

    stype: known / new_osint / new_cold (benign) ; bec_off / bec_fab (attack).
    fab_layer: for bec_fab, the OSINT layer the attacker FAKED an edge in (else None)."""
    rows = []
    tw = list(twins)
    for v in tw:
        c = sorted(contact_nbr.get(v, set()))
        o_only = sorted(osint_union.get(v, set()) - contact_nbr.get(v, set()) - {v})
        off = [t for t in tw if t != v and t not in contact_nbr.get(v, set())
               and t not in osint_union.get(v, set())]
        for k in range(EVENTS_PER_VICTIM):
            h = _h(str(seed), v, str(k))
            if h % 2 == 0:                                    # benign
                r = (h // 3) % 100
                if r < P_KNOWN * 100 and c:
                    rows.append((c[h % len(c)], v, 0, "known", None))
                elif r < (P_KNOWN + P_NEW_OSINT) * 100 and o_only:
                    rows.append((o_only[h % len(o_only)], v, 0, "new_osint", None))
                elif off:
                    rows.append((off[h % len(off)], v, 0, "new_cold", None))    # legit but no tie -> FP risk for ALL
            else:                                              # BEC attack
                if not off:
                    continue
                s = off[h % len(off)]
                if (h // 5) % 100 < int(fabrication_rate * 100) and osint_layers:
                    rows.append((s, v, 1, "bec_fab", osint_layers[h % len(osint_layers)]))  # planted fake footprint
                else:
                    rows.append((s, v, 1, "bec_off", None))
    return rows


def _features(rows, adj, layer_names, victim_map):
    """contact-only X, multiplex X. Honour per-event fabricated OSINT edge (bec_fab)."""
    Xc, Xm = [], []
    for s, v, _y, _st, fab in rows:
        vv = victim_map(v)
        per = []
        for L in layer_names:
            on = (s in adj[L].get(vv, set())) or (L == fab)   # fabricated edge fools the feature
            per.append(1.0 if on else 0.0)
        cov = sum(per[1:])
        Xc.append([per[0]])
        Xm.append(per + [sum(per), cov, 1.0 if cov > 0 else 0.0])
    return np.array(Xc, dtype=np.float32), np.array(Xm, dtype=np.float32)


# Warstwy OSINT z REALNYCH pol blizniaka (c3-c8) — glowny wynik RQ2.
# Syntetyczny overlay (build_overlay) uzywany TYLKO do sweepow adwersaryjnych (p_cross/fabrykacja),
# bo realne warstwy maja stale, naturalne pokrycie cross-org (brak pokretla p_cross).
REAL_OSINT = ["osint_conference", "osint_certification", "osint_skill", "osint_routine",
              "osint_event", "osint_pretext", "osint_platform", "interest_sim"]


def run(p_cross, fabrication_rate, real=False, perseed_out=None):
    base = build_layers()
    contact = {"contact": base["contact"]}
    accum = {k: [] for k in ["contact_auc", "mux_auc", "shuf_auc",
                             "contact_r1", "mux_r1", "shuf_r1",
                             "fp_newsender_contact", "fp_newsender_mux"]}
    for seed in SEEDS:
        if real:
            layers = {**contact, **{L: base[L] for L in REAL_OSINT}}
        else:
            ov, _ = build_overlay(seed=seed, p_cross=p_cross, fabrication_rate=fabrication_rate)
            layers = {**contact, **ov}
        names = list(layers)
        adj = _adj(layers)
        contact_nbr = adj["contact"]
        osint_layers = names[1:]
        osint_union = {}
        for L in osint_layers:
            for t, nb in adj[L].items():
                osint_union.setdefault(t, set()).update(nb)
        twins = sorted({t for t in adj["contact"]} | set(osint_union) |
                       {r["twin_id"] for r in _self_rows()})
        rows = _events(twins, contact_nbr, osint_union, osint_layers, seed, fabrication_rate)
        if not rows:
            continue
        y = np.array([r[2] for r in rows]); stype = np.array([r[3] for r in rows])
        vic = [r[1] for r in rows]
        order = sorted(set(vic))
        wrong = {order[i]: order[(i + 7) % len(order)] for i in range(len(order))}
        te_v = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order)//3)]])
        te = np.array([v in te_v for v in vic]); tr = ~te

        Xc, Xm = _features(rows, adj, names, lambda v: v)
        _, Xs = _features(rows, adj, names, lambda v: wrong[v])

        def fit(X):
            clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=200, verbose=-1).fit(X[tr], y[tr])
            return clf.predict_proba(X[te])[:, 1]

        sc, sm, ss = fit(Xc), fit(Xm), fit(Xs)
        yte = y[te]
        accum["contact_auc"].append(roc_auc_score(yte, sc))
        accum["mux_auc"].append(roc_auc_score(yte, sm))
        accum["shuf_auc"].append(roc_auc_score(yte, ss))
        accum["contact_r1"].append(_recall_at_fpr(yte, sc))
        accum["mux_r1"].append(_recall_at_fpr(yte, sm))
        accum["shuf_r1"].append(_recall_at_fpr(yte, ss))
        # New-sender FP: among legit new-on-OSINT senders, what fraction does each model
        # push into the top-1% (i.e. falsely treats as attack)? Lower is better; multiplex
        # should rescue them (they carry OSINT coverage) where contact-only cannot.
        st_te = stype[te]
        thr_c = np.quantile(sc, 0.99); thr_m = np.quantile(sm, 0.99)
        newm = (yte == 0) & (st_te == "new_osint")
        accum["fp_newsender_contact"].append(float((sc[newm] >= thr_c).mean()) if newm.any() else 0.0)
        accum["fp_newsender_mux"].append(float((sm[newm] >= thr_m).mean()) if newm.any() else 0.0)
        if perseed_out is not None:
            for k, v in accum.items():
                perseed_out.append([seed, k, round(float(v[-1]), 4)])
    means = {k: (float(np.mean(v)) if v else 0.0) for k, v in accum.items()}
    sds = {k + "_sd": (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) for k, v in accum.items()}
    return {**means, **sds}


def _self_rows():
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    out = RESULTS / "exp_multiplex.csv"
    grid = [("default", 0.5, 0.0), ("low_cross", 0.2, 0.0), ("high_cross", 0.8, 0.0),
            ("fab10", 0.5, 0.1), ("fab20", 0.5, 0.2), ("fab50", 0.5, 0.5),
            ("fab75", 0.5, 0.75), ("fab100", 0.5, 1.0)]
    rowsout = []
    perseed = []
    print(f"{'config':11s} | {'AUC contact/mux/shuf':>22s} | {'R@1% contact/mux/shuf':>22s} "
          f"| {'newsndr-FP c/m':>14s}")
    for name, pc, fab in grid:
        # glowny wynik RQ2 ("default") na REALNYCH warstwach; sweepy p_cross/fabrykacji
        # na syntetycznym kontrolowanym wariancie (realne warstwy nie maja tych pokretel).
        _ps = []
        r = run(pc, fab, real=(name == "default"), perseed_out=_ps)
        for row in _ps:
            perseed.append([name, pc, fab] + row)
        rowsout.append({"config": name, "p_cross": pc, "fabrication": fab, **{k: round(v, 3) for k, v in r.items()}})
        print(f"{name:11s} | {r['contact_auc']:6.3f} {r['mux_auc']:6.3f} {r['shuf_auc']:6.3f}     | "
              f"{r['contact_r1']:6.3f} {r['mux_r1']:6.3f} {r['shuf_r1']:6.3f}     | "
              f"{r['fp_newsender_contact']:6.3f} {r['fp_newsender_mux']:6.3f}")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rowsout[0])); w.writeheader(); w.writerows(rowsout)
    print(f"\n[exp-multiplex] wrote {out}")
    out_ps = RESULTS / "exp_multiplex_perseed.csv"
    with out_ps.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "p_cross", "fabrication", "seed", "metric", "value"])
        w.writerows(perseed)


if __name__ == "__main__":
    main()
