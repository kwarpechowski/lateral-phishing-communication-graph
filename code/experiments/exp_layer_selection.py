"""Selekcja warstw (#GP-EXP-9): ile / ktore warstwy sa niezbedne, ktore pomijalne.

Forward greedy selection (dodawaj warstwe o najwiekszym marginalnym AUC) + leave-one-out
z pelnego zestawu (usun warstwe -> spadek AUC). Operuje na PELNYM zestawie warstw:
kontakty + hierarchia + domena + 7 typow OSINT + wspolaktywnosc czasowa + cecha
sp. czasowej (TEMPORAL). Pokazuje minimalny wystarczajacy podzbior i warstwy redundantne.

Reuzywa modelu zdarzen i _eval z exp_extended. Wyjscie: results/exp_layer_selection.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from experiments.exp_extended import SEEDS, _eval, _prepare        # noqa: E402
from graph.build_layers import build_layers                        # noqa: E402
from graph.build_temporal_overlay import coactivity_edges          # noqa: E402

TEMPORAL = "TEMPORAL"   # specjalny kandydat (cecha sp. czasowej per-zdarzenie)


def _augment(adj, twins, seed):
    """Dodaj do adj warstwy strukturalne (hierarchia/domena) + wspolaktywnosc czasowa."""
    base = build_layers()
    for nm in ("org_hierarchy", "shared_domain"):
        nbr = {}
        for a, b in base[nm]:
            nbr.setdefault(a, set()).add(b); nbr.setdefault(b, set()).add(a)
        adj[nm] = nbr
    nbr = {}
    for a, b in coactivity_edges(twins, seed):
        nbr.setdefault(a, set()).add(b); nbr.setdefault(b, set()).add(a)
    adj["temporal_coactivity"] = nbr
    return adj


def _eval_state(per_seed, graph_layers, temporal):
    """Srednie AUC po ziarnach dla stanu (zbior warstw grafowych, flaga temporal)."""
    aucs = []
    for seed, (adj, rows, twins) in per_seed.items():
        subset = graph_layers if graph_layers else ["contact"]
        a, _r = _eval(adj, rows, twins, seed, subset, temporal)
        aucs.append(a)
    return float(np.mean(aucs))


def main():
    # przygotuj dane raz na ziarno
    per_seed = {}
    candidates = None
    for seed in SEEDS:
        adj, names, rows, twins = _prepare(seed, 0.5, 0.0, real=True)   # realne warstwy z c3-c8
        adj = _augment(adj, twins, seed)
        per_seed[seed] = (adj, rows, twins)
        if candidates is None:
            candidates = [k for k in adj] + [TEMPORAL]   # wszystkie warstwy grafowe + temporal

    # --- forward greedy selection ---
    selected_graph, temporal_on = [], False
    remaining = set(candidates)
    order = []
    while remaining:
        best, best_auc = None, -1.0
        for c in remaining:
            if c == TEMPORAL:
                auc = _eval_state(per_seed, selected_graph, True)
            else:
                auc = _eval_state(per_seed, selected_graph + [c], temporal_on)
            if auc > best_auc:
                best, best_auc = c, auc
        order.append((best, round(best_auc, 4)))
        if best == TEMPORAL:
            temporal_on = True
        else:
            selected_graph.append(best)
        remaining.discard(best)
        print(f"  +{best:22s} -> AUC={best_auc:.3f}", flush=True)

    # --- leave-one-out z pelnego zestawu ---
    full_graph = [c for c in candidates if c != TEMPORAL]
    full_auc = _eval_state(per_seed, full_graph, True)
    loo = []
    for c in candidates:
        if c == TEMPORAL:
            auc = _eval_state(per_seed, full_graph, False)
        else:
            auc = _eval_state(per_seed, [x for x in full_graph if x != c], True)
        loo.append((c, round(full_auc - auc, 4)))
    loo.sort(key=lambda x: -x[1])

    out = RESULTS / "exp_layer_selection.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "layer", "value"])
        for i, (lay, auc) in enumerate(order, 1):
            w.writerow(["forward_cumulative_auc", f"{i}:{lay}", auc])
        for lay, drop in loo:
            w.writerow(["leave_one_out_drop", lay, drop])
        w.writerow(["full", "all_layers", round(full_auc, 4)])

    print(f"\n[layer-sel] pelny AUC ({len(candidates)} warstw) = {full_auc:.3f}")
    print("[layer-sel] kolejnosc forward (kumulacyjny AUC):")
    for i, (lay, auc) in enumerate(order, 1):
        print(f"   {i:2d}. {lay:22s} {auc:.3f}")
    print("[layer-sel] leave-one-out (spadek AUC po usunieciu; >0 = niezbedna):")
    for lay, drop in loo:
        flag = "NIEZBEDNA" if drop >= 0.01 else "pomijalna"
        print(f"   {lay:22s} {drop:+.3f}  [{flag}]")
    print(f"[layer-sel] wrote {out}")


if __name__ == "__main__":
    main()
