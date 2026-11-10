"""Trzeci sweep: korelacja SPOLECZNOSC-ETYKIETA przy STALEJ klasteryzacji (recenzja P3 #2).

Zarzut (slusznosc): sweepy 1-2 (modularnosc / planted-clique) NIE tlumacza glownego wyniku, bo graf
referencyjny org-800 ma NAJWYZSZA klasteryzacje, a GCN jest na nim ZAWYZONY. Hipoteza recenzenta:
zawyzenie bierze sie z WYROWNANIA spolecznosci z etykietami (organizacje = spolecznosci = granice kaskad),
nie z samej klasteryzacji.

Test: w planted-clique TRZYMAMY p_clique (gestosc wewnatrz-spolecznosciowa -> ~stala klasteryzacja),
a ZMIENIAMY frakcje krawedzi MIEDZY-spolecznosciowych `inter`. Wysokie inter ROZMYWA granice
spolecznosci (kaskada przelewa sie miedzy spolecznosciami -> SLABNIE wyrownanie spolecznosc-etykieta),
przy ~stalej klasteryzacji. Jesli zawyzenie GCN SPADA z inter (przy stalej klasteryzacji), a temporalny
trzyma -> mechanizmem jest wyrownanie spolecznosc-etykieta, nie klasteryzacja. To domyka sprzecznosc.

Wyjscie: results/exp_p3_community_label.csv  (uruchom z code/).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from cascadebench import CascadeStrategy, Graph, build_scenario, evaluate, get_panel  # noqa: E402
from cascadebench.topology import planted_clique_edgelist, graph_stats, modularity     # noqa: E402

SEEDS = 6
N, K = 600, 6
P_CLIQUE = 0.45                       # STALA gestosc wewnatrz-spolecznosciowa -> ~stala klasteryzacja
INTER_GRID = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]   # rosnace krawedzie miedzy-spolecznosciowe
STRAT = CascadeStrategy(fanout=4, spread=2)


def main():
    tmp = RESULTS.parent / "data" / "topo_tmp"; tmp.mkdir(parents=True, exist_ok=True)
    names = [d.name for d in get_panel()]
    gi = names.index("GCN-statyczny"); ti = names.index("temporalny-GNN")
    rows = []
    print(f"  {'inter':>6s} {'klaster.':>8s} {'modular.':>8s} {'AUC GCN':>8s} {'AUC temp':>8s}")
    for inter in INTER_GRID:
        clusters, mods, gcn_auc, tmp_auc = [], [], [], []
        for s in range(SEEDS):
            p = planted_clique_edgelist(tmp / f"pc_i{inter}_{s}.txt", n=N, k=K,
                                        p_clique=P_CLIQUE, inter=inter, seed=s)
            g = Graph.from_edgelist(p, name=f"pc-i{inter}")
            st = graph_stats(g)
            clusters.append(st["clustering"]); mods.append(modularity(g, K, N))
            ev = evaluate(build_scenario(g, STRAT, s), get_panel(), s)
            gcn_auc.append(ev[names[gi]]["auc"]); tmp_auc.append(ev[names[ti]]["auc"])
        cl, mo = float(np.mean(clusters)), float(np.mean(mods))
        ga, ta = float(np.mean(gcn_auc)), float(np.mean(tmp_auc))
        rows.append([inter, round(cl, 3), round(mo, 3), round(ga, 3), round(ta, 3)])
        print(f"  {inter:6.2f} {cl:8.3f} {mo:8.3f} {ga:8.3f} {ta:8.3f}", flush=True)
    out = RESULTS / "exp_p3_community_label.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["inter", "clustering", "modularity", "auc_GCN", "auc_temporal"])
        w.writerows(rows)
    # podsumowanie korelacji
    from scipy.stats import pearsonr
    inter = [r[0] for r in rows]; cl = [r[1] for r in rows]; mo = [r[2] for r in rows]
    ga = [r[3] for r in rows]; ta = [r[4] for r in rows]
    print(f"\n  klasteryzacja ~ stala (zakres {min(cl):.2f}-{max(cl):.2f})")
    print(f"  AUC GCN vs modularnosc (wyrownanie spol.-etykieta): r={pearsonr(mo, ga)[0]:+.2f}")
    print(f"  AUC GCN vs klasteryzacja: r={pearsonr(cl, ga)[0]:+.2f}")
    print(f"  AUC temporalny vs modularnosc: r={pearsonr(mo, ta)[0]:+.2f}")
    print(f"[community-label] -> {out}")


if __name__ == "__main__":
    main()
