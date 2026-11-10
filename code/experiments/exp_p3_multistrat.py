"""Transfer rang USREDNIONY po siatce strategii atakujacego (recenzja P3 #3).

Zarzut: glowny wynik (transfer rang Spearmana, asymetria GCN/temporalne) zmierzono pod JEDNA
strategia atakujacego, a Tab. 4 pokazuje, ze ranking detektorow odwraca sie z rezimem ataku.
Tu powtarzamy sonde predyktywnosci pod KILKOMA strategiami (naiwna / skryta / niski-zasieg / mimikra)
i raportujemy transfer + luke GCN/temporalna USREDNIONE po strategiach, z rozrzutem.

Wyjscie: results/exp_p3_multistrat.csv  (uruchom z code/).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from cascadebench import CascadeStrategy, build_scenario, evaluate, get_panel  # noqa: E402
from cascadebench.topology import topology_pool                                # noqa: E402

SEEDS = 3
STRATS = {                                   # siatka strategii (zasieg x ewazja) — 3 rezimy z D1
    "naiwna-K8":    CascadeStrategy(fanout=8, spread=1),
    "skryta-K3-g4": CascadeStrategy(fanout=3, spread=4),
    "niski-K2":     CascadeStrategy(fanout=2, spread=2),
}


def main():
    tmp = RESULTS.parent / "data" / "topo_tmp"
    pool = topology_pool(tmp, core_cap=1500)
    names = [d.name for d in get_panel()]
    ref = pool[0][0]
    rows = []
    # per-strategia: transfer rho (synt vs real) + luka GCN/temporalna
    per_strat = {}
    print(f"  {'strategia':14s} {'rho synt':>9s} {'rho real':>9s} {'luka GCN':>9s} {'luka temp':>9s}")
    for sname, strat in STRATS.items():
        auc = {}
        for gname, g, kind in pool:
            mat = [[evaluate(build_scenario(g, strat, s), get_panel(), s)[n]["auc"] for n in names]
                   for s in range(SEEDS)]
            auc[gname] = np.array(mat).mean(0)
        kinds = {g: k for g, _, k in pool}
        rho_synth, rho_real = [], []
        for gname, _, kind in pool[1:]:
            rho = float(spearmanr(auc[ref], auc[gname])[0])
            (rho_real if kind == "real" else rho_synth).append(rho)
        gi = names.index("GCN-statyczny"); ti = names.index("temporalny-GNN")
        synth_g = [g for g, _, k in pool if k in ("org", "rand")]
        real_g = [g for g, _, k in pool if k == "real"]
        gap_gcn = float(np.mean([auc[g][gi] for g in real_g]) - np.mean([auc[g][gi] for g in synth_g]))
        gap_tmp = float(np.mean([auc[g][ti] for g in real_g]) - np.mean([auc[g][ti] for g in synth_g]))
        per_strat[sname] = (np.median(rho_synth), np.median(rho_real), gap_gcn, gap_tmp)
        print(f"  {sname:14s} {np.median(rho_synth):9.2f} {np.median(rho_real):9.2f} "
              f"{gap_gcn:+9.3f} {gap_tmp:+9.3f}", flush=True)
        rows.append([sname, round(float(np.median(rho_synth)), 3), round(float(np.median(rho_real)), 3),
                     round(gap_gcn, 4), round(gap_tmp, 4)])

    # USREDNIENIE po strategiach + rozrzut (zakres)
    arr = np.array([per_strat[s] for s in STRATS])     # rows=strats, cols=(rho_s,rho_r,gap_gcn,gap_tmp)
    mean = arr.mean(0); lo = arr.min(0); hi = arr.max(0)
    print(f"\n  [usrednione po {len(STRATS)} strategiach] "
          f"rho synt={mean[0]:.2f}[{lo[0]:.2f},{hi[0]:.2f}] real={mean[1]:.2f}[{lo[1]:.2f},{hi[1]:.2f}] "
          f"| luka GCN={mean[2]:+.3f}[{lo[2]:+.3f},{hi[2]:+.3f}] temp={mean[3]:+.3f}[{lo[3]:+.3f},{hi[3]:+.3f}]")
    rows.append(["SREDNIA", round(float(mean[0]), 3), round(float(mean[1]), 3),
                 round(float(mean[2]), 4), round(float(mean[3]), 4)])
    rows.append(["ZAKRES_min", round(float(lo[0]), 3), round(float(lo[1]), 3),
                 round(float(lo[2]), 4), round(float(lo[3]), 4)])
    rows.append(["ZAKRES_max", round(float(hi[0]), 3), round(float(hi[1]), 3),
                 round(float(hi[2]), 4), round(float(hi[3]), 4)])
    out = RESULTS / "exp_p3_multistrat.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["strategia", "rho_synt", "rho_real", "luka_GCN", "luka_temp"])
        w.writerows(rows)
    print(f"[multistrat] -> {out}")


if __name__ == "__main__":
    main()
