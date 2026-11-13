"""Exp C3 (paper C, Track 2) - Fabricated triples in a public KG: FB15k-237.

Task: distinguish FABRICATED triples from TRUE ones in FB15k-237 (Toutanova et al.,
272k train triples, 237 relations) using only structural/relational consistency -
no trained embeddings (so the result is about the graph, not a model).

Fabricators (realistic per Prop. fab: existence/count features must have no power):
  marginal      (h, r, t) sampled i.i.d. from the empirical slot marginals
                (head freq, relation freq, tail freq) - indistinguishable by degree
  hr_preserving (h, r) sampled from OBSERVED (h, r) pairs, t from the tail marginal -
                a stronger fabricator that keeps the head-relation context real
                (in the spirit of poisoning attacks on KG embeddings)

Detection features (higher = more genuine; reported AUC uses fabrication score,
so AUC > 0.5 = detection works):
  degree_sum        log-degrees of h and t                     [existence baseline]
  common_neighbors  |N(h) \\ {t}  n  N(t) \\ {h}|               [path consistency]
  multi_rel_support  |relations linking (h,t) besides r|        [co-relation support]
  combined          rank-sum of common_neighbors + multi_rel   [consistency detector]

Metrics: AUC and sensitivity@FPR1% (positive class = fabricated), 5 seeds,
20k true vs 20k fabricated candidates per seed. Pure stdlib.
Output: results/exp_kg_poisoning.csv
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
DATA = CODE / "data" / "fb15k237"
OUT = CODE / "results" / "exp_kg_poisoning.csv"

SEEDS = list(range(10))
N_CAND = 20000
FB15K_URL = (
    "https://raw.githubusercontent.com/TimDettmers/ConvE/master/FB15k-237.tar.gz"
)


def ensure_fb15k() -> None:
    """Download FB15k-237 (~3.7 MB tar.gz, public) if not present."""
    if (DATA / "train.txt").exists():
        return
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"[fb15k-237] downloading {FB15K_URL} -> {DATA}")
    import tarfile
    import urllib.request

    tgz = DATA / "fb15k237.tar.gz"
    urllib.request.urlretrieve(FB15K_URL, tgz)
    with tarfile.open(tgz) as tar:
        for member in tar.getmembers():
            if member.name in ("train.txt", "valid.txt", "test.txt"):
                member.name = member.name  # flatten
                tar.extract(member, DATA)
    print(f"[fb15k-237] extracted train/valid/test to {DATA}")


def load_triples(path: Path):
    triples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def auc(pos: list[float], neg: list[float]) -> float:
    """Rank AUC with tie credit; pos = fabricated (higher score = more suspicious)."""
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
    ensure_fb15k()
    triples = load_triples(DATA / "train.txt")
    print(f"[fb15k-237] {len(triples)} train triples")

    triple_set = set(triples)
    nbr: dict[str, set] = defaultdict(set)
    pair_rels: dict[tuple[str, str], set] = defaultdict(set)
    deg: dict[str, int] = defaultdict(int)
    heads, tails, rels = [], [], []
    hr_pairs = []
    for h, r, t in triples:
        nbr[h].add(t)
        nbr[t].add(h)
        pair_rels[(h, t)].add(r)
        deg[h] += 1
        deg[t] += 1
        heads.append(h)
        tails.append(t)
        rels.append(r)
        hr_pairs.append((h, r))
    entities = sorted(deg)

    def features(h: str, r: str, t: str) -> dict:
        nh = nbr[h] - {t}
        nt = nbr[t] - {h}
        small, big = (nh, nt) if len(nh) <= len(nt) else (nt, nh)
        cn = len(small & big)
        mr = len(pair_rels.get((h, t), set()) - {r})
        return {
            "common_neighbors": cn,
            "multi_rel_support": mr,
            "degree_sum": math.log1p(deg[h]) + math.log1p(deg[t]),
        }

    rows_out = []
    for seed in SEEDS:
        rng = random.Random(seed)
        true = rng.sample(triples, N_CAND)

        fab_marginal = []
        while len(fab_marginal) < N_CAND:
            cand = (rng.choice(heads), rng.choice(rels), rng.choice(tails))
            if cand not in triple_set and cand[0] != cand[2]:
                fab_marginal.append(cand)

        fab_hr = []
        while len(fab_hr) < N_CAND:
            h, r = rng.choice(hr_pairs)
            t = rng.choice(tails)
            cand = (h, r, t)
            if cand not in triple_set and cand[0] != cand[2]:
                fab_hr.append(cand)

        # path-preserving fabricator: the adversary KNOWS the common-neighbour
        # feature and samples t from the 2-hop neighbourhood of h, so the
        # fabricated triple has common neighbours by construction.
        fab_path = []
        while len(fab_path) < N_CAND:
            h = rng.choice(heads)
            direct = nbr[h]
            if not direct:
                continue
            two_hop = set()
            for x in rng.sample(sorted(direct), min(len(direct), 25)):
                two_hop |= nbr[x]
            two_hop -= direct
            two_hop.discard(h)
            if not two_hop:
                continue
            t = rng.choice(sorted(two_hop))
            r = rng.choice(rels)
            cand = (h, r, t)
            if (
                cand not in triple_set
                and cand[0] != cand[2]
                and (h, t) not in pair_rels
            ):
                fab_path.append(cand)

        for fabricator, fab in (
            ("marginal", fab_marginal),
            ("hr_preserving", fab_hr),
            ("path_preserving", fab_path),
        ):
            fets_t = [features(*tr) for tr in true]
            fets_f = [features(*tr) for tr in fab]
            for fname in ("degree_sum", "common_neighbors", "multi_rel_support"):
                # fabrication score = -feature (higher = more suspicious)
                pos = [-f[fname] for f in fets_f]
                neg = [-f[fname] for f in fets_t]
                rows_out.append(
                    {
                        "dataset": "fb15k237",
                        "fabricator": fabricator,
                        "seed": seed,
                        "feature": fname,
                        "auc": round(auc(pos, neg), 4),
                        "sens_fpr1": round(sens_at_fpr(pos, neg), 4),
                    }
                )
            # combined: rank-sum of the two consistency features
            all_cn = [f["common_neighbors"] for f in fets_t] + [
                f["common_neighbors"] for f in fets_f
            ]
            all_mr = [f["multi_rel_support"] for f in fets_t] + [
                f["multi_rel_support"] for f in fets_f
            ]

            def ranks(vals):
                order = sorted(range(len(vals)), key=lambda i: vals[i])
                rk = [0.0] * len(vals)
                i = 0
                while i < len(vals):
                    j = i
                    while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                        j += 1
                    for k in range(i, j + 1):
                        rk[order[k]] = (i + j) / 2
                    i = j + 1
                return rk

            rk_cn, rk_mr = ranks(all_cn), ranks(all_mr)
            comb = [-(a + b) for a, b in zip(rk_cn, rk_mr)]
            pos, neg = comb[N_CAND:], comb[:N_CAND]
            rows_out.append(
                {
                    "dataset": "fb15k237",
                    "fabricator": fabricator,
                    "seed": seed,
                    "feature": "combined",
                    "auc": round(auc(pos, neg), 4),
                    "sens_fpr1": round(sens_at_fpr(pos, neg), 4),
                }
            )
        print(f"  seed {seed} done")

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"[exp_kg_poisoning] {len(rows_out)} rows -> {OUT.name}")
    agg: dict[tuple, list] = defaultdict(list)
    agg_s: dict[tuple, list] = defaultdict(list)
    for r in rows_out:
        agg[(r["fabricator"], r["feature"])].append(r["auc"])
        agg_s[(r["fabricator"], r["feature"])].append(r["sens_fpr1"])
    for key in sorted(agg):
        print(
            f"  {key[0]:<14} {key[1]:<18} "
            f"AUC={sum(agg[key]) / len(agg[key]):.3f} "
            f"sens@1%={sum(agg_s[key]) / len(agg_s[key]):.3f}"
        )

    # ---------- paired contrasts over per-seed AUCs ----------
    def wilcoxon_p(a: list[float], b: list[float]) -> float:
        """Wilcoxon signed-rank, normal approximation (two-sided)."""
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

    def cliffs_delta_paired(a: list[float], b: list[float]) -> float:
        """Paired dominance: P(a>b) + 0.5 P(a=b), in [-1, 1]."""
        n = len(a)
        s = sum(1 if x > y else (-1 if x < y else 0) for x, y in zip(a, b))
        return s / n

    contrast_rows = []
    for fabricator in ("marginal", "hr_preserving", "path_preserving"):
        for contrast, f1, f2 in (
            ("combined_vs_degree", "combined", "degree_sum"),
            ("combined_vs_multirel", "combined", "multi_rel_support"),
        ):
            a = [
                r["auc"]
                for r in rows_out
                if r["fabricator"] == fabricator and r["feature"] == f1
            ]
            b = [
                r["auc"]
                for r in rows_out
                if r["fabricator"] == fabricator and r["feature"] == f2
            ]
            contrast_rows.append(
                {
                    "contrast": contrast,
                    "fabricator": fabricator,
                    "dAUC": round(sum(x - y for x, y in zip(a, b)) / len(a), 4),
                    "wilcoxon_p": round(wilcoxon_p(a, b), 5),
                    "cliffs_d": round(cliffs_delta_paired(a, b), 2),
                    "n_seeds": len(a),
                }
            )
    OUT_C = CODE / "results" / "exp_kg_contrasts.csv"
    with OUT_C.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(contrast_rows[0].keys()))
        w.writeheader()
        w.writerows(contrast_rows)
    print(f"[exp_kg_poisoning] {len(contrast_rows)} contrasts -> {OUT_C.name}")
    for r in contrast_rows:
        print(
            f"  {r['fabricator']:<14} {r['contrast']:<20} "
            f"dAUC={r['dAUC']:+.3f} p={r['wilcoxon_p']:.4f} d={r['cliffs_d']:+.2f}"
        )


if __name__ == "__main__":
    main()
