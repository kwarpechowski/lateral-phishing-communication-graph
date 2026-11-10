"""Czy redundancja warstw strukturalnych to artefakt TOPOLOGII? (#GP-EXP-30, recenzja P1 pkt 7).

Zarzut: Jaccard 0.9+ miedzy contact/hierarchia/domena to bezposrednia konsekwencja generatora
(19 firm x ~8 osob, jedna domena na firme, geste kontakty wewnatrzfirmowe). W realnej organizacji
warstwy te bylyby mniej skorelowane i moglyby niesc NIEZALEZNY sygnal. Testujemy to wprost
TANIM eksperymentem na syntetyku: wprowadzamy pokretlo DEKORELACJI topologii.

Pokretlo rho in [0,1]: frakcja krawedzi kontaktow PRZEPISANYCH z wewnatrz-firmowych na
MIEDZY-firmowe (kontakt przestaje pokrywac sie z granica firmy). Warstwy hierarchia/domena
pozostaja firmowe. Im wyzsze rho, tym mniejsza korelacja contact~hierarchia~domena.

Dla kazdego rho mierzymy (20 ziaren):
  * jaccard_struct : srednia parami Jaccard{contact, hierarchia, domena} (korelacja warstw),
  * full_auc       : AUC pelnego multipleksu strukturalnego (contact+hierarchia+domena),
  * contact_auc    : AUC samego kontaktu (baseline),
  * loo_struct     : przyrost AUC warstw strukturalnych nad kontaktem (full - contact)
                     = NIEZALEZNY sygnal warstw hierarchia/domena.

Hipoteza: przy rho=0 (oryginal) warstwy redundantne (loo_struct ~ 0); wraz ze wzrostem rho
Jaccard spada, a loo_struct rosnie -> redundancja jest artefaktem topologii, nie metody.

Zdarzenia: benign-znany (kontakt), benign-kolega (nie-kontakt, lecz hierarchia/domena),
atak (w zadnej z trzech warstw). Pure-local; uruchom z code/. Wyjscie:
results/exp_topology_correlation.csv
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from graph.build_layers import (_domain_edges, _load_twins,  # noqa: E402
                                _org_hierarchy_edges, _self_table, _contact_edges)

SEEDS = list(range(20))
RHOS = [0.0, 0.2, 0.4, 0.6, 0.8]
EVENTS_PER_VICTIM = 16


def _h(*p: str) -> int:
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def _adj(edges):
    nbr = defaultdict(set)
    for a, b in edges:
        nbr[a].add(b); nbr[b].add(a)
    return nbr


def _decorrelate(contact, org_of, twins, rho, seed):
    """Przepisz frakcje rho krawedzi kontaktu wewnatrz-firmowych na miedzy-firmowe.

    Utrzymuje liczbe krawedzi; jeden koniec losowany do twina z INNEJ firmy -> contact
    przestaje pokrywac sie z klika firmowa (spada Jaccard z hierarchia/domena)."""
    by_org = defaultdict(list)
    for t in twins:
        by_org[org_of[t]].append(t)
    out = set()
    cl = sorted(contact)
    for i, (a, b) in enumerate(cl):
        h = _h("rho", str(seed), str(i))
        if (h % 1000) / 1000.0 < rho:
            # przepnij koniec b na twina z innej firmy
            others = [t for t in twins if org_of[t] != org_of[a] and t != a]
            if others:
                nb = others[h % len(others)]
                out.add(tuple(sorted((a, nb))))
                continue
        out.add((a, b))
    return out


def _jaccard(ea, eb):
    ea, eb = set(ea), set(eb)
    return len(ea & eb) / len(ea | eb) if (ea or eb) else 0.0


def _events(contact_nbr, struct_union, twins, seed):
    """benign-znany / benign-kolega(struktura) / atak(brak), label 0/0/1."""
    rows = []
    tw = list(twins)
    for v in tw:
        c = sorted(contact_nbr.get(v, set()))
        only_struct = sorted(struct_union.get(v, set()) - contact_nbr.get(v, set()) - {v})
        off = [t for t in tw if t != v and t not in contact_nbr.get(v, set())
               and t not in struct_union.get(v, set())]
        if not c or not off:
            continue
        for k in range(EVENTS_PER_VICTIM):
            h = _h(str(seed), v, str(k)); r = h % 3
            if r == 0:
                rows.append((c[h % len(c)], v, 0))                         # znany kontakt
            elif r == 1 and only_struct:
                rows.append((only_struct[h % len(only_struct)], v, 0))     # kolega (hierarchia/domena)
            else:
                rows.append((off[h % len(off)], v, 1))                     # atak: w zadnej warstwie
    return rows


def _run(contact, hierarchy, domain, org_of, twins, rho, seed):
    cc = _decorrelate(contact, org_of, twins, rho, seed)
    layers = {"contact": _adj(cc), "hierarchy": _adj(hierarchy), "domain": _adj(domain)}
    struct_union = defaultdict(set)
    for L in ("hierarchy", "domain"):
        for t, nb in layers[L].items():
            struct_union[t].update(nb)
    rows = _events(layers["contact"], struct_union, twins, seed)
    if not rows:
        return None
    y = np.array([r[2] for r in rows]); vic = [r[1] for r in rows]
    order = sorted(set(vic))
    te_v = set(np.array(order)[np.random.default_rng(seed).permutation(len(order))[:max(1, len(order)//3)]])
    te = np.array([v in te_v for v in vic]); tr = ~te

    def feat(use_struct):
        X = []
        for s, v, _l in rows:
            con = 1.0 if s in layers["contact"].get(v, set()) else 0.0
            if not use_struct:
                X.append([con]); continue
            hi = 1.0 if s in layers["hierarchy"].get(v, set()) else 0.0
            dm = 1.0 if s in layers["domain"].get(v, set()) else 0.0
            X.append([con, hi, dm, con + hi + dm])
        return np.array(X, np.float32)

    def fit(X):
        clf = LGBMClassifier(n_jobs=1, random_state=42, n_estimators=150, verbose=-1).fit(X[tr], y[tr])
        return roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])

    full = fit(feat(True)); con = fit(feat(False))
    j = np.mean([_jaccard(cc, hierarchy), _jaccard(cc, domain), _jaccard(hierarchy, domain)])
    return {"jaccard_struct": j, "full_auc": full, "contact_auc": con, "loo_struct": full - con}


def main():
    twins_d, self_tbl = _load_twins(), _self_table()
    twins = sorted(self_tbl)
    org_of = {t: self_tbl[t]["org"] for t in twins}
    contact = _contact_edges()
    hierarchy = _org_hierarchy_edges(self_tbl)
    domain = _domain_edges(self_tbl, twins_d)
    print(f"[topo] {len(twins)} twins; contact={len(contact)} hierarchy={len(hierarchy)} domain={len(domain)}")
    rows = []
    print(f"  {'rho':>4s} {'Jaccard':>8s} {'full':>6s} {'contact':>8s} {'loo_struct':>10s}")
    for rho in RHOS:
        acc = defaultdict(list)
        for seed in SEEDS:
            r = _run(contact, hierarchy, domain, org_of, twins, rho, seed)
            if r:
                for k, v in r.items():
                    acc[k].append(v)
        agg = {k: float(np.mean(v)) for k, v in acc.items()}
        rows.append([rho, round(agg["jaccard_struct"], 3), round(agg["full_auc"], 3),
                     round(agg["contact_auc"], 3), round(agg["loo_struct"], 3)])
        print(f"  {rho:4.1f} {agg['jaccard_struct']:8.3f} {agg['full_auc']:6.3f} "
              f"{agg['contact_auc']:8.3f} {agg['loo_struct']:10.3f}")
    out = RESULTS / "exp_topology_correlation.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["rho", "jaccard_struct", "full_auc", "contact_auc", "loo_struct"])
        w.writerows(rows)
    print(f"[topo] -> {out}")


if __name__ == "__main__":
    main()
