"""cascadebench.search — automatyczny red/blue co-search (AutoML ataku i obrony).

Trzy tryby:
  attack_search   — atakujacy-OPTYMALIZATOR: szuka strategii MINIMALIZUJACEJ detekcje przy zasiegu
                    >= reach_min (adaptacyjny atak w przestrzeni problemu).
  defense_search  — wybiera NAJODPORNIEJSZY detektor wobec danej strategii ataku.
  arms_race       — wyscig zbrojen: naprzemienna optymalizacja atak<->obrona, trajektoria do rownowagi.
To odroznia poligon od statycznych benchmarkow: nie ustalone eksperymenty, lecz AUTOMATYCZNE
odkrywanie najsilniejszego ataku i najodporniejszej obrony.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from . import (CascadeStrategy, build_scenario, victim_split, get_panel, TemporalGNN, Detector)

RESULTS = Path(__file__).resolve().parents[1] / "results"

# przestrzen strategii atakujacego (pokretla zasiegu i ewazji)
SPACE = {"fanout": [2, 3, 5, 8], "spread": [1, 2, 4, 8], "mimicry": [0.0, 0.3, 0.6]}


def evaluate_strategy(graph, strat: CascadeStrategy, detector: Detector,
                      seeds: int = 2) -> Tuple[float, float]:
    """Srednia (AUC detekcji, zasieg) dla strategii vs detektor."""
    aucs, reach = [], []
    for s in range(seeds):
        sc = build_scenario(graph, strat, s)
        tr, te = victim_split(sc.events, s)
        yte = np.array([e[3] for e in sc.events], float)[te]
        aucs.append(roc_auc_score(yte, detector.fit_score(sc, tr, te, s)))
        reach.append(sc.infected_frac)
    return float(np.mean(aucs)), float(np.mean(reach))


def _sample(rng) -> CascadeStrategy:
    return CascadeStrategy(fanout=int(rng.choice(SPACE["fanout"])),
                           spread=int(rng.choice(SPACE["spread"])),
                           mimicry=float(rng.choice(SPACE["mimicry"])))


def _mutate(strat: CascadeStrategy, rng) -> CascadeStrategy:
    key = rng.choice(list(SPACE))
    val = rng.choice(SPACE[key])
    kw = {"fanout": strat.fanout, "spread": strat.spread, "mimicry": strat.mimicry}
    kw[key] = int(val) if key != "mimicry" else float(val)
    return CascadeStrategy(**kw)


def attack_search(graph, detector: Optional[Detector] = None, reach_min: float = 0.5,
                  budget: int = 12, seeds: int = 2, seed: int = 0, verbose: bool = True):
    """Ewolucyjne szukanie strategii o NAJNIZSZEJ detekcji przy zasiegu >= reach_min.
    Zwraca (best_strategy, best_auc, best_reach), history."""
    detector = detector or TemporalGNN()
    rng = np.random.default_rng(seed)
    history = []
    best = None  # (strat, score, auc, reach)
    cur = _sample(rng)
    for it in range(budget):
        strat = cur if it == 0 else _mutate(best[0], rng)
        auc, reach = evaluate_strategy(graph, strat, detector, seeds)
        feasible = reach >= reach_min
        score = auc + (0.0 if feasible else 10.0)            # kara za niewystarczajacy zasieg
        history.append((strat.label(), auc, reach, feasible))
        if best is None or score < best[1]:
            best = (strat, score, auc, reach)
        if verbose:
            print(f"  iter {it}: {strat.label()} det={auc:.3f} reach={reach:.0%} "
                  f"{'OK' if feasible else 'za maly zasieg'}", flush=True)
    return (best[0], best[2], best[3]), history


def defense_search(graph, strat: CascadeStrategy, detectors: Optional[List[Detector]] = None,
                   seeds: int = 2) -> Tuple[str, Dict[str, float]]:
    """Najodporniejszy detektor (max AUC) wobec danej strategii ataku."""
    detectors = detectors or get_panel()
    res = {d.name: evaluate_strategy(graph, strat, d, seeds)[0] for d in detectors}
    return max(res, key=res.get), res


def transferability(graph: str = "synthetic:300", detectors_subset=None, reach_min: float = 0.5,
                    budget: int = 4, search_seeds: int = 1, eval_seeds: int = 2,
                    seed: int = 0, save: bool = True):
    """Transferowalnosc atakow: optymalizuj atak vs KAZDY detektor, oceniaj vs WSZYSTKIE.
    Macierz [atak-zoptymalizowany-vs-wiersz][oceniany-vs-kolumna]. Diagonala = white-box;
    poza diagonala = transfer. Niska poza diagonala = detektor specyficzny (trudniejszy do ataku przenoszonego).
    """
    from . import load
    g = load(graph)
    panel = get_panel()
    dets = [d for d in panel if d.name in detectors_subset] if detectors_subset else panel
    names = [d.name for d in dets]
    # 1) najlepszy atak przeciw kazdemu detektorowi (white-box)
    best = {}
    for d in dets:
        print(f"=== optymalizacja ataku vs {d.name} ===", flush=True)
        (strat, _auc, _reach), _ = attack_search(g, d, reach_min, budget, search_seeds, seed, verbose=False)
        best[d.name] = strat
        print(f"  najlepszy atak vs {d.name}: {strat.label()}", flush=True)
    # 2) macierz transferu: atak(wiersz) oceniany vs detektor(kolumna)
    matrix = []
    for an in names:
        row = [round(evaluate_strategy(g, best[an], d, eval_seeds)[0], 3) for d in dets]
        matrix.append([an] + row)
        print(f"  atak[{an}] vs " + " ".join(f"{n}={v:.2f}" for n, v in zip(names, row)), flush=True)
    if save:
        out = RESULTS / "cb_transfer.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["atak\\detektor"] + names); w.writerows(matrix)
        print(f"[cb] -> {out}")
    return names, matrix


# rozszerzona, drobniejsza przestrzen dla SILNIEJSZEGO adwersarza (white-box best-response)
STRONG_SPACE = {"fanout": [1, 2, 3, 5], "spread": [2, 4, 8, 16],
                "mimicry": [0.3, 0.6, 1.0], "fabrication": [0.0, 0.3]}


def _sample_strong(rng) -> CascadeStrategy:
    return CascadeStrategy(fanout=int(rng.choice(STRONG_SPACE["fanout"])),
                           spread=int(rng.choice(STRONG_SPACE["spread"])),
                           mimicry=float(rng.choice(STRONG_SPACE["mimicry"])),
                           fabrication=float(rng.choice(STRONG_SPACE["fabrication"])))


def _mutate_strong(strat: CascadeStrategy, rng) -> CascadeStrategy:
    key = rng.choice(list(STRONG_SPACE)); val = rng.choice(STRONG_SPACE[key])
    kw = {"fanout": strat.fanout, "spread": strat.spread, "mimicry": strat.mimicry,
          "fabrication": strat.fabrication}
    kw[key] = int(val) if key == "fanout" or key == "spread" else float(val)
    return CascadeStrategy(**kw)


def strong_adversary(graph: str = "synthetic:600", budget: int = 24, seeds: int = 3,
                     reach_min: float = 0.4, seed: int = 0, save: bool = True):
    """SILNIEJSZY adwersarz: white-box best-response przeciw wyuczonemu temporalnemu GNN nad
    ROZSZERZONA przestrzenia akcji w przestrzeni problemu (pelna mimikra rytmu, max rozproszenie,
    fabrykacja, fan-out=1). Odpowiedz na zarzut, ze przestrzen ataku jest zbyt zgrubna.
    Raportuje: (1) najsilniejsza ewazja vs temporalny GNN (AUC + Recall@FPR1%) i jej zasieg,
    (2) transfer tej ewazji na CALY panel, (3) porownanie z atakiem nieadaptacyjnym (naiwnym)."""
    from . import load, get_panel, recall_at_fpr, aggregate, evaluate
    g = load(graph); target = TemporalGNN()
    rng = np.random.default_rng(seed)
    # 1) ewolucyjny white-box search vs temporalny GNN (minimalizuj AUC przy zasiegu >= reach_min)
    best = None; cur = _sample_strong(rng)
    print(f"=== SILNIEJSZY ADWERSARZ: white-box vs {target.name}, budzet {budget} ===", flush=True)
    for it in range(budget):
        strat = cur if it == 0 else _mutate_strong(best[0], rng)
        auc, reach = evaluate_strategy(g, strat, target, seeds)
        score = auc + (0.0 if reach >= reach_min else 10.0)
        if best is None or score < best[1]:
            best = (strat, score, auc, reach)
        print(f"  iter {it}: {strat.label()} det(GNN)={auc:.3f} reach={reach:.0%}", flush=True)
    bstrat, _, b_auc, b_reach = best
    print(f"\n  >> najsilniejsza ewazja: {bstrat.label()} | temporalny GNN AUC={b_auc:.3f}, zasieg={b_reach:.0%}", flush=True)
    # 2) transfer najlepszej ewazji na caly panel (AUC + Recall@FPR1%); 3) baseline naiwny
    names = [d.name for d in get_panel()]
    def panel_metrics(strat, label):
        per = [evaluate(build_scenario(g, strat, s), get_panel(), s) for s in range(seeds)]
        agg = aggregate(per)
        row_auc = [round(agg[n]["auc"], 3) for n in names]
        row_r1 = [round(agg[n]["recall@0.01"], 3) for n in names]
        print(f"  [{label}] AUC: " + " ".join(f"{n}={v:.2f}" for n, v in zip(names, row_auc)), flush=True)
        return row_auc, row_r1
    naive = CascadeStrategy(fanout=8, spread=1)
    na_auc, na_r1 = panel_metrics(naive, "nieadaptacyjny (K8)")
    ev_auc, ev_r1 = panel_metrics(bstrat, f"white-box ({bstrat.label()})")
    if save:
        out = RESULTS / "cb_strong_adversary.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["wariant", "metryka"] + names)
            w.writerow(["nieadaptacyjny", "auc"] + na_auc); w.writerow(["nieadaptacyjny", "recall_fpr1"] + na_r1)
            w.writerow([f"white-box:{bstrat.label()}", "auc"] + ev_auc)
            w.writerow([f"white-box:{bstrat.label()}", "recall_fpr1"] + ev_r1)
        print(f"[cb] -> {out}")
    return {"best": bstrat.label(), "gnn_auc": b_auc, "reach": b_reach,
            "naive_auc": dict(zip(names, na_auc)), "evasive_auc": dict(zip(names, ev_auc))}


def joint_panel_adversary(graph: str = "synthetic:600", budget: int = 24, seeds: int = 3,
                          reach_min: float = 0.4, seed: int = 0, save: bool = True):
    """Atak WIELOKRYTERIALNY: minimalizuje MAKSYMALNE AUC po CALYM panelu (najlepszy detektor),
    nie pojedynczy temporalny GNN. Wprost testuje zarzut recenzenta adversarial ML: obrona zespolowa
    jest tak mocna, jak najlepszy WSPOLNY atak. Kontrast z D3b (atak vs jeden kanal, sygnal ucieka
    do innego). Raportuje: czy jeden atak tlumi CALY panel jednoczesnie i jakim kosztem zasiegu."""
    from . import load, get_panel, aggregate, evaluate
    g = load(graph)
    rng = np.random.default_rng(seed)
    names = [d.name for d in get_panel()]

    def panel_eval(strat, n_seeds):
        scs = [build_scenario(g, strat, s) for s in range(n_seeds)]
        per = [evaluate(scs[s], get_panel(), s) for s in range(n_seeds)]
        agg = aggregate(per)
        aucs = {n: float(agg[n]["auc"]) for n in names}
        r1 = {n: float(agg[n]["recall@0.01"]) for n in names}
        reach = float(np.mean([sc.infected_frac for sc in scs]))
        return aucs, r1, reach

    print(f"=== ATAK WIELOKRYTERIALNY: minimalizuj MAX AUC po panelu ({len(names)} det.), budzet {budget} ===", flush=True)
    best = None; cur = _sample_strong(rng)
    for it in range(budget):
        strat = cur if it == 0 else _mutate_strong(best[0], rng)
        aucs, r1, reach = panel_eval(strat, seeds)
        maxauc = max(aucs.values())                          # najlepszy detektor = cel do zbicia
        score = maxauc + (0.0 if reach >= reach_min else 10.0)
        if best is None or score < best[1]:
            best = (strat, score, aucs, r1, reach)
        leader = max(aucs, key=aucs.get)
        print(f"  iter {it}: {strat.label()} maxAUC={maxauc:.3f} ({leader}) reach={reach:.0%}", flush=True)
    bstrat, _, b_aucs, b_r1, b_reach = best
    leader = max(b_aucs, key=b_aucs.get)
    print(f"\n  >> najlepszy wspolny atak: {bstrat.label()} | MAX AUC po panelu={b_aucs[leader]:.3f} "
          f"({leader}), zasieg={b_reach:.0%}", flush=True)
    print("  per-detektor AUC: " + " ".join(f"{n}={b_aucs[n]:.2f}" for n in names), flush=True)
    # odniesienie: atak naiwny K8 (pelny zasieg, bez adaptacji)
    na_aucs, na_r1, na_reach = panel_eval(CascadeStrategy(fanout=8, spread=1), seeds)
    print(f"  [nieadaptacyjny K8] MAX AUC={max(na_aucs.values()):.3f}, zasieg={na_reach:.0%}", flush=True)
    if save:
        out = RESULTS / "cb_joint_panel.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["wariant", "metryka", "reach"] + names)
            w.writerow(["nieadaptacyjny", "auc", round(na_reach, 3)] + [round(na_aucs[n], 3) for n in names])
            w.writerow([f"joint:{bstrat.label()}", "auc", round(b_reach, 3)] + [round(b_aucs[n], 3) for n in names])
            w.writerow([f"joint:{bstrat.label()}", "recall_fpr1", round(b_reach, 3)] + [round(b_r1[n], 3) for n in names])
        print(f"[cb] -> {out}")
    return {"best": bstrat.label(), "max_auc": b_aucs[leader], "leader": leader, "reach": b_reach,
            "panel_auc": b_aucs, "naive_max_auc": max(na_aucs.values()), "naive_reach": na_reach}


def arms_race(graph: str = "synthetic:400", rounds: int = 3, reach_min: float = 0.5,
              budget: int = 6, seeds: int = 2, seed: int = 0, save: bool = True):
    """Wyscig zbrojen: atak optymalizuje vs obecna obrona, obrona wybiera najlepszy detektor vs atak."""
    from . import load
    g = load(graph)
    name2det = {d.name: d for d in get_panel()}
    detector = TemporalGNN(); traj = []
    for r in range(rounds):
        print(f"=== runda {r}: atakujacy optymalizuje vs {detector.name} ===", flush=True)
        (strat, atk_auc, reach), _ = attack_search(g, detector, reach_min, budget, seeds, seed + r)
        best_def, det_res = defense_search(g, strat, get_panel(), seeds)
        traj.append([r, strat.label(), round(atk_auc, 4), round(reach, 4), best_def,
                     round(det_res[best_def], 4)])
        print(f"  -> najlepszy atak {strat.label()} (detekcja {atk_auc:.3f}, zasieg {reach:.0%}); "
              f"najodporniejsza obrona: {best_def} ({det_res[best_def]:.3f})", flush=True)
        detector = name2det[best_def]                        # nast. runda: atak vs najlepsza obrona
    if save:
        out = RESULTS / "cb_arms_race.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["runda", "atak", "detekcja_auc", "zasieg",
                                           "najlepsza_obrona", "obrona_auc"])
            w.writerows(traj)
        print(f"[cb] -> {out}")
    return traj
