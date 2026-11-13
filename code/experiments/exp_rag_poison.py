"""Exp C2 (paper C, Track 1) - GraphRAG-lite corpus poisoning and the consistency gate.

Setting: a note-graph retrieval in the spirit of GraphRAG. The knowledge base is the
twins multiplex rendered as natural-language notes: internal layers (contacts /
hierarchy / domain - NOT fabricatable from outside) and OSINT layers (mailing lists,
conferences, OSS projects, ... - public facts).

Attack (PoisonedRAG-style): a cold attacker (a node outside every layer) injects a
poisoned note claiming a shared OSINT community with each victim. Two styles:
  naive      full claim sentence ("X shares conference with attacker")
  optimized  retrieval-optimized minimal text ("X shares") - the PoisonedRAG recipe
Escalation k = number of ADDITIONAL fabricated OSINT memberships the attacker plants
in other categories (each planted fact = an independent public trace with its own cost).

Defense - consistency gate: a retrieved note about pair (a,b) is trusted only if the
pair has internal support (contact/hierarchy/domain/co-activity) OR at least two
independent OSINT categories (the claimed fact + >=1 independent corroboration).

Metrics (top-5, victim queries, 5 seeds):
  corruption@5   fraction of attacked victims with >=1 poisoned note in top-5
                 (defense off vs on), per escalation k in {0,1,2}
  clean cost     fraction of victims losing a genuine cross-org single-fact OSINT tie
                 (the paper-B "legitimate new sender") from top-5 due to the gate

TF-IDF is hand-rolled (pure stdlib, deterministic). Output: results/exp_rag_poison.csv
"""

from __future__ import annotations

import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

from graph.build_osint_overlay import build_overlay, CATEGORIES  # noqa: E402

RESULTS = CODE / "results"
LAYERS = RESULTS / "layers"
OUT = RESULTS / "exp_rag_poison.csv"

SEEDS = [0, 1, 2, 3, 4]
TOPK = 5
STRUCTURAL = {
    "contact": 1.0,
    "org_hierarchy": 0.9,
    "shared_domain": 0.8,
    "temporal_coactivity": 0.5,
}
GATE_STRUCT_MIN = 0.5
GATE_OSINT_EXTRA_MIN = 1

CATEGORY_LABEL = {
    "osint_mailing_list": "mailing list",
    "osint_oss_project": "open-source project",
    "osint_affiliation": "professional affiliation",
    "osint_vendor": "vendor platform",
    "osint_certification": "certification",
    "osint_event": "industry event",
    "osint_social": "social channel",
}


def _tokens(text: str) -> list[str]:
    return [t for t in text.lower().replace(",", " ").split() if t]


class TfIdf:
    """Minimal TF-IDF + cosine; deterministic."""

    def __init__(self, docs: list[str]):
        self.N = len(docs)
        self.doc_tokens = [_tokens(d) for d in docs]
        df: dict[str, int] = defaultdict(int)
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] += 1
        self.idf = {t: math.log((1 + self.N) / (1 + d)) + 1 for t, d in df.items()}

    def _vec(self, toks: list[str]) -> dict[str, float]:
        v: dict[str, float] = defaultdict(float)
        for t in toks:
            v[t] += self.idf.get(t, math.log(1 + self.N) + 1)
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, query: str) -> list[float]:
        qv = self._vec(_tokens(query))
        sims = []
        for toks in self.doc_tokens:
            dv = self._vec(toks)
            sims.append(sum(w * dv.get(t, 0.0) for t, w in qv.items()))
        return sims


def load_structural_support() -> dict[tuple[str, str], float]:
    supp: dict[tuple[str, str], float] = defaultdict(float)
    for layer, w in STRUCTURAL.items():
        with (LAYERS / f"{layer}.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                a, b = row["twin_a"], row["twin_b"]
                supp[(a, b)] += w
                supp[(b, a)] += w
    return supp


def load_self() -> dict[str, dict[str, str]]:
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        return {r["twin_id"]: r for r in csv.DictReader(f)}


def _h(*p: str) -> int:
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def main() -> None:
    names = load_self()
    twins = sorted(names)
    supp = load_structural_support()
    rows_out = []

    def gate_ok(pair: tuple[str, str], extra_osint: int) -> bool:
        if supp.get(pair, 0.0) >= GATE_STRUCT_MIN:
            return True
        return extra_osint >= GATE_OSINT_EXTRA_MIN

    for seed in SEEDS:
        layers, _ = build_overlay(seed, 0.5, 0.0)
        cats_of: dict[tuple[str, str], set] = defaultdict(set)
        for cat, edges in layers.items():
            for a, b in edges:
                cats_of[(a, b)].add(cat)
                cats_of[(b, a)].add(cat)

        # ---------- clean corpus ----------
        docs: list[str] = []
        meta: list[dict] = []
        for (a, b), cats in sorted(cats_of.items()):
            if a > b:
                continue
            cat = sorted(cats)[0]
            docs.append(
                f"{names[a]['name']} shares {CATEGORY_LABEL[cat]} with {names[b]['name']}"
            )
            meta.append(
                {
                    "pair": (a, b),
                    "kind": "osint",
                    "cross": names[a]["org"] != names[b]["org"],
                }
            )
        for layer, phrase in (
            ("contact", "corresponds with"),
            ("org_hierarchy", "works with at"),
            ("shared_domain", "shares mail domain with"),
        ):
            with (LAYERS / f"{layer}.csv").open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    a, b = r["twin_a"], r["twin_b"]
                    if a > b:
                        continue
                    docs.append(f"{names[a]['name']} {phrase} {names[b]['name']}")
                    meta.append({"pair": (a, b), "kind": "internal", "cross": False})
        n_clean = len(docs)
        index = TfIdf(docs)

        # ---------- clean cost of the gate ----------
        cross_lost = 0
        victims_with_cross = 0
        for v in twins:
            q = f"{names[v]['name']} shares"
            sims = index.search(q)
            order = sorted(range(n_clean), key=lambda i: -sims[i])
            top = [
                i
                for i in order[:TOPK]
                if meta[i]["pair"][0] == v or meta[i]["pair"][1] == v
            ]
            cross_before = [
                i
                for i in top
                if meta[i]["kind"] == "osint"
                and meta[i]["cross"]
                and len(cats_of[meta[i]["pair"]]) == 1
            ]
            if not cross_before:
                continue
            victims_with_cross += 1
            if any(not gate_ok(meta[i]["pair"], 0) for i in cross_before):
                cross_lost += 1
        rows_out.append(
            {
                "seed": seed,
                "esc": "clean",
                "style": "",
                "defend": "",
                "policy": "",
                "corruption": "",
                "n_attempts": "",
                "cross_lost_victims": cross_lost,
                "victims_with_cross": victims_with_cross,
            }
        )

        # ---------- attack: style x escalation x defense x policy ----------
        for k in (0, 1, 2):
            for style in ("naive", "optimized"):
                for defend in (0, 1):
                    for policy in ("skip", "demote"):
                        if not defend and policy == "demote":
                            continue  # identical to skip when defense is off
                        corrupted = 0
                        attempts = 0
                        for attacker in [f"attacker-{j}" for j in range(3)]:
                            for v in twins:
                                if _h(str(seed), attacker, v) % 4 != 0:
                                    continue
                                attempts += 1
                                pdocs, pmeta = list(docs), list(meta)
                                claim_cat = sorted(CATEGORIES)[
                                    _h(str(seed), attacker, v, "cat") % len(CATEGORIES)
                                ]
                                if style == "naive":
                                    pdocs.append(
                                        f"{names[v]['name']} shares "
                                        f"{CATEGORY_LABEL[claim_cat]} with {attacker}"
                                    )
                                else:
                                    pdocs.append(f"{names[v]['name']} shares")
                                pmeta.append({"pair": (v, attacker), "kind": "poison"})
                                for j in range(k):
                                    corr_cat = sorted(CATEGORIES)[
                                        _h(str(seed), attacker, v, "corr", str(j))
                                        % len(CATEGORIES)
                                    ]
                                    if corr_cat == claim_cat:
                                        corr_cat = sorted(CATEGORIES)[
                                            (sorted(CATEGORIES).index(corr_cat) + 1)
                                            % len(CATEGORIES)
                                        ]
                                    if style == "naive":
                                        pdocs.append(
                                            f"{names[v]['name']} shares "
                                            f"{CATEGORY_LABEL[corr_cat]} with {attacker}"
                                        )
                                    else:
                                        pdocs.append(f"{names[v]['name']} shares")
                                    pmeta.append(
                                        {"pair": (v, attacker), "kind": "poison"}
                                    )
                                pindex = TfIdf(pdocs)
                                q = f"{names[v]['name']} shares"
                                sims = pindex.search(q)
                                if defend and policy == "demote":
                                    # demote gated notes instead of skipping them
                                    sims = [
                                        0.25 * s
                                        if pmeta[i]["kind"] != "internal"
                                        and not gate_ok(
                                            pmeta[i]["pair"],
                                            k
                                            if pmeta[i]["pair"][1] == attacker
                                            else max(
                                                0,
                                                len(cats_of.get(pmeta[i]["pair"], ()))
                                                - 1,
                                            ),
                                        )
                                        else s
                                        for i, s in enumerate(sims)
                                    ]
                                order = sorted(
                                    range(len(pdocs)), key=lambda i: -sims[i]
                                )
                                top, seen = [], 0
                                for i in order:
                                    if (
                                        pmeta[i]["pair"][0] != v
                                        and pmeta[i]["pair"][1] != v
                                    ):
                                        continue
                                    pair = pmeta[i]["pair"]
                                    extra = (
                                        k
                                        if pair[1] == attacker
                                        else max(0, len(cats_of.get(pair, ())) - 1)
                                    )
                                    if (
                                        defend
                                        and policy == "skip"
                                        and pmeta[i]["kind"] != "internal"
                                        and not gate_ok(pair, extra)
                                    ):
                                        continue
                                    top.append(i)
                                    seen += 1
                                    if seen >= TOPK:
                                        break
                                if any(pmeta[i]["kind"] == "poison" for i in top):
                                    corrupted += 1
                        rows_out.append(
                            {
                                "seed": seed,
                                "esc": f"k={k}",
                                "style": style,
                                "defend": defend,
                                "policy": policy,
                                "corruption": round(corrupted / attempts, 4),
                                "n_attempts": attempts,
                                "cross_lost_victims": "",
                                "victims_with_cross": "",
                            }
                        )

    fn = [
        "seed",
        "esc",
        "style",
        "defend",
        "policy",
        "corruption",
        "n_attempts",
        "cross_lost_victims",
        "victims_with_cross",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows_out)
    print(f"[exp_rag_poison] {len(rows_out)} rows -> {OUT.name}")
    agg = defaultdict(list)
    for r in rows_out:
        if r["esc"] != "clean":
            agg[(r["esc"], r["style"], r["defend"], r["policy"])].append(r["corruption"])
    for (esc, style, d, pol), v in sorted(agg.items()):
        print(f"  {esc:<6} {style:<10} defend={d} policy={pol:<7} corruption@5={sum(v) / len(v):.3f}")
    cl = [
        r["cross_lost_victims"] / r["victims_with_cross"]
        for r in rows_out
        if r["esc"] == "clean"
    ]
    print(
        f"  clean cost: cross-org single-fact tie gated for {sum(cl) / len(cl):.1%} "
        f"of victims (mean over seeds)"
    )


if __name__ == "__main__":
    main()
