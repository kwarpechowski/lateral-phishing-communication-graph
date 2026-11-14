"""Exp C1 (paper C, Track 1) — Detecting fabricated OSINT edges by cross-layer consistency.

Task: on the twins multiplex, distinguish FABRICATED OSINT edges from GENUINE ones.
Realistic fabricator (Prop. fab / H9d): fabricated memberships are drawn i.i.d. from the
SAME community pools as genuine ones (build_osint_overlay), so existence/degree features
have no power by construction; the signal must come from consistency with the
STRUCTURAL layers (contact / hierarchy / domain / temporal co-activity) that the
attacker cannot forge from outside.

Feature per candidate edge (a,b):
  struct_support   sum of weights of structural layers containing (a,b)   [main detector]
  degree_sum       OSINT-multiplex degrees of a and b                      [existence baseline]
  osint_multiplicity  number of OSINT categories in which (a,b) is an edge [count baseline]
  struct_shuffled  struct_support values permuted across candidates        [permutation control]

Metrics: AUC (rank/Mann-Whitney) and sensitivity@FPR1% (positive class = fabricated).
Pure stdlib; deterministic given seeds. Output: results/exp_fab_edge_detection.csv
"""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

from graph.build_osint_overlay import build_overlay  # noqa: E402

RESULTS = CODE / "results"
LAYERS = RESULTS / "layers"
OUT = RESULTS / "exp_fab_edge_detection.csv"

SEEDS = list(range(10))
FAB_RATES = [0.05, 0.10, 0.20, 0.30, 0.50]

# structural layers the attacker cannot easily forge (internal / behavioural)
STRUCTURAL = {
    "contact": 1.0,
    "org_hierarchy": 0.9,
    "shared_domain": 0.8,
    "temporal_coactivity": 0.5,
}


def _h(*p: str) -> int:
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def load_structural_support() -> dict[tuple[str, str], float]:
    supp: dict[tuple[str, str], float] = defaultdict(float)
    for layer, w in STRUCTURAL.items():
        path = LAYERS / f"{layer}.csv"
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                a, b = row["twin_a"], row["twin_b"]
                supp[(a, b)] += w
                supp[(b, a)] += w
    return supp


def auc(pos: list[float], neg: list[float]) -> float:
    """Rank-based AUC (Mann-Whitney U), ties get 0.5 credit."""
    if not pos or not neg:
        return float("nan")
    vals = [(v, lab) for lab, group in ((1, pos), (0, neg)) for v in group]
    vals.sort(key=lambda x: x[0])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1][0] == vals[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    s = sum(rk for rk, (_, lab) in zip(ranks, vals) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)


def sens_at_fpr(pos: list[float], neg: list[float], fpr: float = 0.01) -> float:
    """Max sensitivity s.t. FPR <= fpr (score higher = more 'fabricated')."""
    if not pos or not neg:
        return float("nan")
    budget = max(1, int(fpr * len(neg)))  # max allowed false positives
    cands = sorted(set(neg) | {max(neg) + 1}, reverse=True)
    best_s = 0.0
    for t in cands:
        fp = sum(1 for v in neg if v >= t)
        if fp > budget:
            break  # fp only grows as t decreases
        best_s = max(best_s, sum(1 for v in pos if v >= t) / len(pos))
    return best_s


def main() -> None:
    supp = load_structural_support()
    rows_out = []
    for seed in SEEDS:
        for fab in FAB_RATES:
            layers, _ = build_overlay(seed, 0.5, fab)
            # union of candidate edges; genuine = ANY category reports genuine
            genuine: dict[tuple[str, str], bool] = {}
            cats_of: dict[tuple[str, str], set] = defaultdict(set)
            for cat, edges in layers.items():
                for (a, b), gen in edges.items():
                    key = (a, b)
                    cats_of[key].add(cat)
                    genuine[key] = genuine.get(key, False) or gen
            # OSINT-multiplex degree (existence feature)
            deg: dict[str, int] = defaultdict(int)
            for a, b in genuine:
                deg[a] += 1
                deg[b] += 1
            keys = sorted(genuine)
            feats = {
                k: {}
                for k in (
                    "struct_support",
                    "degree_sum",
                    "osint_multiplicity",
                    "struct_shuffled",
                )
            }
            for k in keys:
                a, b = k
                feats["struct_support"][k] = -supp.get(k, 0.0)
                feats["degree_sum"][k] = -(deg[a] + deg[b])
                feats["osint_multiplicity"][k] = -len(cats_of[k])
            # permutation control: hash-assignment of struct_support values
            vals = [feats["struct_support"][k] for k in keys]
            for i, k in enumerate(keys):
                j = _h(str(seed), str(fab), "perm", str(i)) % len(keys)
                feats["struct_shuffled"][k] = vals[j]
            pos_keys = [k for k in keys if not genuine[k]]
            neg_keys = [k for k in keys if genuine[k]]
            for fname, fdict in feats.items():
                pos = [fdict[k] for k in pos_keys]
                neg = [fdict[k] for k in neg_keys]
                rows_out.append(
                    {
                        "seed": seed,
                        "fab_rate": fab,
                        "feature": fname,
                        "auc": round(auc(pos, neg), 4),
                        "sens_fpr1": round(sens_at_fpr(pos, neg), 4),
                        "n_fab": len(pos_keys),
                        "n_gen": len(neg_keys),
                    }
                )
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"[exp_fab_edge_detection] {len(rows_out)} rows -> {OUT.name}")
    # quick digest
    agg: dict[tuple, list] = defaultdict(list)
    for r in rows_out:
        agg[(r["fab_rate"], r["feature"])].append(r["auc"])
    for (fab, feat), a in sorted(agg.items()):
        print(f"  fab={fab:<5} {feat:<20} AUC={sum(a) / len(a):.3f}")

    # ---------- paired contrasts: struct_support vs degree_sum per fab rate ----------
    def wilcoxon_p(a: list[float], b: list[float]) -> float:
        import math

        d = [x - y for x, y in zip(a, b)]
        d = [x for x in d if x != 0]
        n = len(d)
        if n == 0:
            return 1.0
        ranked = sorted(range(n), key=lambda i: abs(d[i]))
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and abs(d[ranked[j + 1]]) == abs(d[ranked[i]]):
                j += 1
            avg = (i + j) / 2 + 1
            for k2 in range(i, j + 1):
                r[ranked[k2]] = avg
            i = j + 1
        wplus = sum(rk for rk, di in zip(r, d) if di > 0)
        mu = n * (n + 1) / 4
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        if sigma == 0:
            return 1.0
        z = (wplus - mu) / sigma
        return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    contrast_rows = []
    for fab in FAB_RATES:
        a = [
            r["auc"]
            for r in rows_out
            if str(r["fab_rate"]) == str(fab) and r["feature"] == "struct_support"
        ]
        b = [
            r["auc"]
            for r in rows_out
            if str(r["fab_rate"]) == str(fab) and r["feature"] == "degree_sum"
        ]
        contrast_rows.append(
            {
                "contrast": "struct_support_vs_degree",
                "fab_rate": fab,
                "dAUC": round(sum(x - y for x, y in zip(a, b)) / len(a), 4),
                "wilcoxon_p": round(wilcoxon_p(a, b), 5),
                "n_seeds": len(a),
            }
        )
    OUT_C = RESULTS / "exp_fab_contrasts.csv"
    with OUT_C.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(contrast_rows[0].keys()))
        w.writeheader()
        w.writerows(contrast_rows)
    print(f"[exp_fab_edge_detection] {len(contrast_rows)} contrasts -> {OUT_C.name}")
    for r in contrast_rows:
        print(f"  fab={r['fab_rate']:<5} dAUC={r['dAUC']:+.3f} p={r['wilcoxon_p']:.4f}")


if __name__ == "__main__":
    main()
