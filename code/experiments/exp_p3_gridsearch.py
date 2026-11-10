"""Wyczerpujacy GRID-SEARCH adwersarza vs panel (recenzja P3 #7).

Zarzut: wniosek o "wspoldzielonej podatnosci panelu" opiera sie na EWOLUCYJNYM optymalizatorze
(24 iteracje, 3 ziarna, ~3 pokretla) — przy tak malej przestrzeni akcji wyczerpujacy grid-search jest
wykonalny i usuwa zarzut, ze rownowaga to artefakt budzetu.

Tu enumerujemy CALA siatke (fanout x spread x mimicry) i znajdujemy atak minimalizujacy MAKSYMALNE
AUC po panelu (przy ograniczeniu zasiegu reach>=reach_min). Porownujemy z wynikiem ewolucyjnym.

Wyjscie: results/exp_p3_gridsearch.csv  (uruchom z code/).
"""
from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from cascadebench import CascadeStrategy, build_scenario, evaluate, aggregate, get_panel, load  # noqa: E402

GRAPH = "synthetic:600"
SEEDS = 3
REACH_MIN = 0.4
FANOUT = [2, 3, 5, 8]
SPREAD = [1, 2, 4, 8]
MIMICRY = [0.0, 0.3, 0.6]


def main():
    g = load(GRAPH)
    names = [d.name for d in get_panel()]

    def panel_eval(strat):
        scs = [build_scenario(g, strat, s) for s in range(SEEDS)]
        agg = aggregate([evaluate(scs[s], get_panel(), s) for s in range(SEEDS)])
        aucs = {n: float(agg[n]["auc"]) for n in names}
        reach = float(np.mean([sc.infected_frac for sc in scs]))
        return aucs, reach

    grid = list(itertools.product(FANOUT, SPREAD, MIMICRY))
    print(f"=== GRID-SEARCH: {len(grid)} strategii x {SEEDS} ziaren, panel {len(names)} det. ===", flush=True)
    best = None; rows = []
    for (K, gp, mm) in grid:
        strat = CascadeStrategy(fanout=K, spread=gp, mimicry=mm)
        aucs, reach = panel_eval(strat)
        maxauc = max(aucs.values())
        feasible = reach >= REACH_MIN
        score = maxauc + (0.0 if feasible else 10.0)
        leader = max(aucs, key=aucs.get)
        rows.append([f"K{K}-g{gp}-m{mm}", round(reach, 3), round(maxauc, 3), leader,
                     int(feasible)] + [round(aucs[n], 3) for n in names])
        if feasible and (best is None or maxauc < best[1]):
            best = (strat, maxauc, aucs, reach, leader)
    bstrat, bmax, baucs, breach, bleader = best
    print(f"\n  >> GRID-optymalny (wyczerpujacy): {bstrat.label()} | MAX AUC po panelu={bmax:.3f} "
          f"({bleader}), zasieg={breach:.0%}", flush=True)
    print("     per-detektor: " + " ".join(f"{n}={baucs[n]:.2f}" for n in names), flush=True)
    # ile strategii zbija MAX AUC ponizej progu? (czy jakikolwiek atak omija CALY panel)
    feas = [r for r in rows if r[4] == 1]
    minmax = min(r[2] for r in feas)
    print(f"  Najnizsze MAX-AUC po panelu wsrod {len(feas)} dopuszczalnych: {minmax:.3f} "
          f"-> {'zaden atak nie omija panelu (MAX AUC wysoko)' if minmax >= 0.7 else 'UWAGA: atak omija panel'}", flush=True)
    out = RESULTS / "exp_p3_gridsearch.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strategia", "reach", "max_auc_panel", "leader", "feasible"] + names)
        w.writerows(sorted(rows, key=lambda r: r[2]))   # od najnizszego MAX AUC
    print(f"[gridsearch] -> {out}")


if __name__ == "__main__":
    main()
