"""cascadebench.experiments — reprodukowalne eksperymenty poligonu (czyste API zamiast ad-hoc skryptow).

  pareto      — front ewazja vs zasieg (sweep fan-out K)
  adaptive    — adaptacyjny atak przy STALYM zasiegu (sweep rozproszenia g)
  panel       — panel detektorow pod strategiami atakujacego
  leak_audit  — audyt wycieku rytmu (sweep benign off-hours)
Kazdy zwraca liste wierszy i (opcjonalnie) zapisuje CSV do results/.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from . import (Graph, load, CascadeStrategy, build_scenario, get_panel, evaluate, aggregate,
               TemporalGNN, victim_split, recall_at_fpr, shuffle_time)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _write(name: str, header: List[str], rows: List[list]):
    out = RESULTS / f"cb_{name}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"[cb] -> {out}")
    return out


def pareto(graph: str = "synthetic:600", seeds: int = 3,
           ks=(1, 2, 3, 5, 8, 999), save: bool = True) -> List[list]:
    g = load(graph); det = TemporalGNN(); rows = []
    for K in ks:
        aucs, infe = [], []
        for s in range(seeds):
            sc = build_scenario(g, CascadeStrategy(fanout=K), s)
            tr, te = victim_split(sc.events, s)
            yte = np.array([e[3] for e in sc.events], float)[te]
            sco = det.fit_score(sc, tr, te, s)
            aucs.append(roc_auc_score(yte, sco)); infe.append(sc.infected_frac)
        rows.append([K, round(float(np.mean(aucs)), 4), round(float(np.mean(infe)), 4)])
        print(f"  K={K}: AUC={rows[-1][1]:.3f} zasieg={rows[-1][2]:.0%}", flush=True)
    if save:
        _write("pareto", ["fanout", "auc", "infected_frac"], rows)
    return rows


def adaptive(graph: str = "synthetic:600", seeds: int = 3, fanout: int = 5,
             gaps=(1, 2, 3, 5, 8), mimicry=(0.0, 0.5), save: bool = True) -> List[list]:
    g = load(graph); det = TemporalGNN(); rows = []
    for m in mimicry:
        for gp in gaps:
            aucs, infe = [], []
            for s in range(seeds):
                sc = build_scenario(g, CascadeStrategy(fanout=fanout, spread=gp, mimicry=m), s)
                tr, te = victim_split(sc.events, s)
                yte = np.array([e[3] for e in sc.events], float)[te]
                aucs.append(roc_auc_score(yte, det.fit_score(sc, tr, te, s)))
                infe.append(sc.infected_frac)
            rows.append([gp, m, round(float(np.mean(aucs)), 4), round(float(np.mean(infe)), 4)])
            print(f"  g={gp} m={m}: AUC={rows[-1][2]:.3f} zasieg={rows[-1][3]:.0%}", flush=True)
    if save:
        _write("adaptive", ["gap", "mimicry", "auc", "infected_frac"], rows)
    return rows


def panel(graph: str = "synthetic:600", seeds: int = 3,
          strategies=None, save: bool = True) -> List[list]:
    strategies = strategies or [("naiwna", CascadeStrategy(fanout=8, spread=1)),
                                ("stealth", CascadeStrategy(fanout=8, spread=8)),
                                ("niski-zasieg", CascadeStrategy(fanout=2, spread=1))]
    g = load(graph); names = [d.name for d in get_panel()]; rows = []
    for sname, strat in strategies:
        per, infe = [], []
        for s in range(seeds):
            sc = build_scenario(g, strat, s)
            per.append(evaluate(sc, get_panel(), s)); infe.append(sc.infected_frac)
        agg = aggregate(per)
        rows.append([sname, round(float(np.mean(infe)), 3)] + [round(agg[n]["auc"], 4) for n in names])
        print(f"  [{sname}] zasieg {np.mean(infe):.0%}: " +
              " ".join(f"{n}={agg[n]['auc']:.2f}" for n in names), flush=True)
    if save:
        _write("panel", ["strategia", "zasieg"] + names, rows)
    return rows


def robustness_curves(graph: str = "synthetic:400", seeds: int = 2,
                      ks=(2, 3, 5, 8), save: bool = True) -> List[list]:
    """Profil odpornosci KAZDEGO detektora: AUC vs zasiegu (fan-out K). Standardowy output benchmarku
    (który detektor degraduje najszybciej, gdy atakujacy schodzi z zasiegiem/wolumenem)."""
    g = load(graph); names = [d.name for d in get_panel()]; rows = []
    for K in ks:
        per, infe = [], []
        for s in range(seeds):
            sc = build_scenario(g, CascadeStrategy(fanout=K), s)
            per.append(evaluate(sc, get_panel(), s)); infe.append(sc.infected_frac)
        agg = aggregate(per)
        rows.append([K, round(float(np.mean(infe)), 3)] + [round(agg[n]["auc"], 4) for n in names])
        print(f"  K={K} (zasieg {np.mean(infe):.0%}): " +
              " ".join(f"{n}={agg[n]['auc']:.2f}" for n in names), flush=True)
    if save:
        _write("robustness_curves", ["fanout", "zasieg"] + names, rows)
    return rows


def validate_real(graphs=("synthetic:600", "email-Eu-core", "CollegeMsg"), seeds: int = 2,
                  strat: Optional[CascadeStrategy] = None, save: bool = True) -> List[list]:
    """Walidacja: czy ranking detektorow z poligonu UTRZYMUJE sie na realnych topologiach?
    Liczy korelacje rang (Spearman) syntetyk vs kazdy realny graf."""
    from scipy.stats import spearmanr
    strat = strat or CascadeStrategy(fanout=8, spread=4)
    names = [d.name for d in get_panel()]
    table = {}
    for gspec in graphs:
        g = load(gspec)
        per = [evaluate(build_scenario(g, strat, s), get_panel(), s) for s in range(seeds)]
        agg = aggregate(per)
        table[gspec] = [round(agg[n]["auc"], 4) for n in names]
        print(f"  [{gspec}] " + " ".join(f"{n}={v:.2f}" for n, v in zip(names, table[gspec])), flush=True)
    ref_spec = graphs[0]; ref = table[ref_spec]
    print(f"\n  korelacja rang detektorow ({ref_spec} vs real):")
    corr_rows = []
    for gspec in graphs[1:]:
        rho, p = spearmanr(ref, table[gspec])
        corr_rows.append([ref_spec, gspec, round(float(rho), 3), round(float(p), 3)])
        print(f"    {gspec}: Spearman rho={rho:.2f} (p={p:.3f})", flush=True)
    if save:
        _write("validate_real", ["graph"] + names, [[g] + table[g] for g in graphs])
        _write("validate_corr", ["ref", "graph", "spearman_rho", "p"], corr_rows)
    return corr_rows


def predictivity_probe(seeds: int = 8, strat: Optional[CascadeStrategy] = None,
                       tmpdir: Optional[str] = None, core_cap: int = 1500, save: bool = True) -> dict:
    """SONDA PREDYKTYWNOSCI (rdzen P3). Panel na zroznicowanej puli topologii (org / ER-BA-WS / realne SNAP).
    Liczy, z PER-ZIARNOWA wariancja (CI):
      (M3) luka transferu syntetyk->real per detektor: mean +/- CI po ziarnach,
      (M1) ANALIZA WRAZLIWOSCI: luka osobno vs org / vs losowe / vs real (czy headline zalezy od skladu puli),
      (M2) KORELACJA AUC detektora z KLASTERYZACJA grafu (czy zawyzenie ~ klasteryzacja generatora),
      (-)  transfer rang (Spearman) + uczciwy negatyw: dystans strukturalny NIE przewiduje transferu.
    """
    from scipy.stats import spearmanr, pearsonr
    from .topology import topology_pool, graph_stats

    strat = strat or CascadeStrategy(fanout=3, spread=3)
    tmp = tmpdir or (RESULTS.parent / "data" / "topo_tmp")
    names = [d.name for d in get_panel()]
    pool = topology_pool(tmp, core_cap=core_cap)
    kinds = {gname: kind for gname, _, kind in pool}

    # 1) PER-ZIARNO wektory AUC + statystyki per topologia
    auc_seed, stats = {}, {}                       # auc_seed[g] = array(seeds x dets)
    for gname, g, kind in pool:
        mat = []
        for s in range(seeds):
            ev = evaluate(build_scenario(g, strat, s), get_panel(), s)
            mat.append([ev[n]["auc"] for n in names])
        auc_seed[gname] = np.array(mat)
        stats[gname] = graph_stats(g)
        mean_v = auc_seed[gname].mean(0)
        print(f"  [{gname}/{kind}] n={stats[gname]['n']:.0f} gini={stats[gname]['deg_gini']:.2f} "
              f"clust={stats[gname]['clustering']:.2f} | best={names[int(np.argmax(mean_v))]}", flush=True)

    def subset(kind):
        return [g for g, _, k in pool if k == kind]
    ORG, RAND, REAL = subset("org"), subset("rand"), subset("real")
    SYNTH = ORG + RAND

    # 2) (M3+M1) luka transferu per detektor: PER-ZIARNO gap = mean(real) - mean(podzbior), potem mean+/-CI
    def gaps_for(synth_names):
        # zwraca slownik det -> (mean_gap, ci95) po ziarnach
        out = {}
        for di, d in enumerate(names):
            per_seed = []
            for s in range(seeds):
                sy = np.mean([auc_seed[g][s, di] for g in synth_names])
                re = np.mean([auc_seed[g][s, di] for g in REAL])
                per_seed.append(re - sy)
            per_seed = np.array(per_seed)
            # kwantyl Studenta (nie 1.96) — uczciwy przy malym n; bootstrap dostepny w stats.mean_ci
            from scipy.stats import t as _t
            ci = (float(_t.ppf(0.975, len(per_seed) - 1)) * per_seed.std(ddof=1) / np.sqrt(len(per_seed))
                  if len(per_seed) > 1 else 0.0)
            out[d] = (float(per_seed.mean()), float(ci))
        return out
    g_all, g_org, g_rand = gaps_for(SYNTH), gaps_for(ORG), gaps_for(RAND)

    # srednie AUC do tabel
    synth_mean = {d: float(np.mean([auc_seed[g][:, di].mean() for g in SYNTH])) for di, d in enumerate(names)}
    real_mean = {d: float(np.mean([auc_seed[g][:, di].mean() for g in REAL])) for di, d in enumerate(names)}

    print("\n  [M1/M3] luka transferu syntetyk->real (mean +/-CI95) | vs org | vs losowe:")
    gap_rows = []
    for d in names:
        ga, ci = g_all[d]; go, _ = g_org[d]; gr, _ = g_rand[d]
        gap_rows.append([d, round(synth_mean[d], 4), round(real_mean[d], 4),
                         round(ga, 4), round(ci, 4), round(go, 4), round(gr, 4)])
        print(f"    {d:22s} {ga:+.3f} +/-{ci:.3f} | org {go:+.3f} | rand {gr:+.3f}", flush=True)

    # 3) (M2) korelacja AUC detektora z klasteryzacja grafu (po wszystkich grafach, srednie po ziarnach)
    clust = [stats[g]["clustering"] for g, _, _ in pool]
    print("\n  [M2] korelacja AUC detektora z KLASTERYZACJA grafu (zawyzenie ~ gestosc lokalna):")
    clust_rows = []
    for di, d in enumerate(names):
        aucs = [auc_seed[g][:, di].mean() for g, _, _ in pool]
        r, p = pearsonr(clust, aucs)
        clust_rows.append([d, round(float(r), 3), round(float(p), 3)])
        print(f"    {d:22s} r={r:+.2f} (p={p:.3f})", flush=True)

    # 4) transfer rang vs ref + uczciwy negatyw (dystans strukturalny NIE przewiduje transferu)
    ref = pool[0][0]; ref_vec = auc_seed[ref].mean(0)
    keys = ["mean_deg", "deg_gini", "clustering", "density"]
    rng_norm = {k: (max(stats[g][k] for g, _, _ in pool) - min(stats[g][k] for g, _, _ in pool)) or 1.0
                for k in keys}
    tr_rows = []
    for gname, _, kind in pool[1:]:
        rho, _ = spearmanr(ref_vec, auc_seed[gname].mean(0))
        dist = sum(abs(stats[gname][k] - stats[ref][k]) / rng_norm[k] for k in keys) / len(keys)
        tr_rows.append([gname, kind, round(float(rho), 3), round(float(dist), 3),
                        round(stats[gname]["deg_gini"], 3), round(stats[gname]["clustering"], 3)])
    pr, pp = pearsonr([r[3] for r in tr_rows], [r[2] for r in tr_rows])
    print(f"\n  [transfer~dystans] Pearson r={pr:.2f} (p={pp:.3f}) "
          f"-> {'BRAK' if pp > 0.05 else 'JEST'} istotna zaleznosc (zalezne od liczby realnych kotwic)", flush=True)

    if save:
        _write("predictivity", ["graph", "kind"] + names + keys,
               [[g, kinds[g]] + [round(auc_seed[g][:, di].mean(), 4) for di in range(len(names))]
                + [round(stats[g][k], 4) for k in keys] for g, _, _ in pool])
        _write("predictivity_gap", ["detektor", "synth_mean", "real_mean", "gap", "gap_ci95", "gap_org", "gap_rand"],
               gap_rows)
        _write("predictivity_clustering", ["detektor", "pearson_r", "p"], clust_rows)
        _write("predictivity_transfer", ["graph", "kind", "rank_transfer", "struct_dist", "deg_gini", "clustering"],
               tr_rows)
        _write("predictivity_finding", ["predictor", "pearson_r", "pearson_p"],
               [["struct_dist", round(float(pr), 3), round(float(pp), 3)]])
    return {"gap": gap_rows, "clustering": clust_rows, "struct_neg": (pr, pp)}


def _reduced_panel(names=("COMPA", "GCN-statyczny", "temporalny-GNN")):
    return [d for d in get_panel() if d.name in names]


def fabrication_sweep(graph: str = "synthetic:600", seeds: int = 3,
                      fabs=(0.0, 0.25, 0.5, 0.75), save: bool = True) -> List[list]:
    """(D) Atak fabrykujacy krawedzie (kontakty spoofowane OSINT, poza znanym grafem). Sweep frakcji
    fabrykacji; mierzy detekcje panelu. Pytanie: czy operowanie POZA grafem kontaktow lamie detektory
    strukturalne (1-hop/GCN/kontekst), czy backfiruje (nowy odbiorca = flagowalny przez COMPA)?"""
    g = load(graph); names = [d.name for d in get_panel()]; rows = []
    for fb in fabs:
        per, infe = [], []
        for s in range(seeds):
            sc = build_scenario(g, CascadeStrategy(fanout=6, spread=2, fabrication=fb), s)
            per.append(evaluate(sc, get_panel(), s)); infe.append(sc.infected_frac)
        agg = aggregate(per)
        rows.append([fb, round(float(np.mean(infe)), 3)] + [round(agg[n]["auc"], 4) for n in names])
        print(f"  fab={fb:.2f} (zasieg {np.mean(infe):.0%}): " +
              " ".join(f"{n}={agg[n]['auc']:.2f}" for n in names), flush=True)
    if save:
        _write("fabrication", ["fabrication", "zasieg"] + names, rows)
    return rows


def mechanism_ablation(seeds: int = 3, n: int = 600, k: int = 6,
                       tmpdir: Optional[str] = None, save: bool = True) -> List[list]:
    """(C) Mechanizm zawyzenia GCN: rozdziela MODULARNOSC od GESTOSCI-WEWNATRZ-SPOLECZNOSCI (klasteryzacji).
    Sweep 1 (SBM): rosnie Q przy NISKIEJ klasteryzacji. Sweep 2 (kliki): rosnie klasteryzacja+Q razem.
    Test rozstrzygajacy: jesli GCN rosnie w sweepie klik, a NIE w SBM -> mechanizm = lokalna gestosc
    (struktura klik/dzialow), nie sama modularnosc. Mierzy GCN vs temporalny vs COMPA."""
    from scipy.stats import pearsonr
    from .topology import sbm_edgelist, planted_clique_edgelist, modularity, graph_stats
    tmp = Path(tmpdir or (RESULTS.parent / "data" / "topo_tmp")); tmp.mkdir(parents=True, exist_ok=True)
    panel = _reduced_panel(); names = [d.name for d in panel]; rows = []

    def run(family, g):
        Q = modularity(g, k, n); clust = graph_stats(g)["clustering"]
        per = [evaluate(build_scenario(g, CascadeStrategy(fanout=3, spread=3), s), panel, s)
               for s in range(seeds)]
        agg = aggregate(per)
        rows.append([family, round(Q, 3), round(clust, 3)] + [round(agg[nm]["auc"], 4) for nm in names])
        print(f"  [{family}] Q={Q:.2f} clust={clust:.2f}: " +
              " ".join(f"{nm}={agg[nm]['auc']:.2f}" for nm in names), flush=True)

    for fi in (0.30, 0.50, 0.70, 0.85, 0.95):            # SBM: modularnosc przy niskiej klasteryzacji
        p = sbm_edgelist(tmp / f"sbm_{int(fi*100)}.txt", n=n, k=k, f_intra=fi, seed=0)
        run("SBM", Graph.from_edgelist(p, name=f"SBM-{fi}"))
    for pc in (0.10, 0.25, 0.45, 0.70):                  # kliki: klasteryzacja+modularnosc razem
        p = planted_clique_edgelist(tmp / f"clq_{int(pc*100)}.txt", n=n, k=k, p_clique=pc, seed=0)
        run("clique", Graph.from_edgelist(p, name=f"CLQ-{pc}"))

    # analiza: korelacja AUC z Q (w SBM) i z klasteryzacja (calosc)
    print("\n  [MECHANIZM] korelacje AUC:")
    corr = []
    sbm = [r for r in rows if r[0] == "SBM"]; allr = rows
    for i, nm in enumerate(names):
        rq, pq = pearsonr([r[1] for r in sbm], [r[3 + i] for r in sbm])         # vs Q (SBM, niska klast.)
        rc, pc2 = pearsonr([r[2] for r in allr], [r[3 + i] for r in allr])      # vs klasteryzacja (calosc)
        corr.append([nm, round(float(rq), 3), round(float(pq), 3), round(float(rc), 3), round(float(pc2), 3)])
        print(f"    {nm:18s} vs Q(SBM): r={rq:+.2f}(p={pq:.3f}) | vs klast(all): r={rc:+.2f}(p={pc2:.3f})",
              flush=True)
    if save:
        _write("mechanism", ["family", "modularity_Q", "clustering"] + names, rows)
        _write("mechanism_corr", ["detektor", "r_vs_Q_sbm", "p_Q", "r_vs_clust_all", "p_clust"], corr)
    return rows


def robustness_grid(seeds_list=(4, 8), cores=(600, 1000), strats=None,
                    save: bool = True) -> List[list]:
    """(A) Siatka robustnosci: czy headline'owe findingi sa STABILNE wzgledem wyborow projektowych?
    Dla kazdej komorki (#ziaren x rdzen x strategia) liczy 3 metryki: luka GCN, luka temporalnego,
    korelacja transfer~dystans. Raportuje, ktore findingi sa stabilne, a ktore kruche."""
    strats = strats or [("low", CascadeStrategy(fanout=3, spread=3)),
                        ("stealth", CascadeStrategy(fanout=3, spread=6, mimicry=0.3))]
    rows = []
    for sd in seeds_list:
        for cc in cores:
            for sname, strat in strats:
                r = predictivity_probe(seeds=sd, core_cap=cc, strat=strat, save=False)
                gap = {d[0]: d[3] for d in r["gap"]}
                pr, pp = r["struct_neg"]
                rows.append([sd, cc, sname,
                             round(gap.get("GCN-statyczny", 0), 3),
                             round(gap.get("temporalny-GNN", 0), 3),
                             round(float(pr), 3), round(float(pp), 3)])
                print(f"  [seeds={sd} core={cc} {sname}] GCN_gap={rows[-1][3]:+.3f} "
                      f"temp_gap={rows[-1][4]:+.3f} struct_r={rows[-1][5]:+.2f}(p={rows[-1][6]:.2f})",
                      flush=True)
    # stabilnosc: znak i zakres
    def col(i): return [r[i] for r in rows]
    gcn = col(3); tmp = col(4); sr = col(5)
    print(f"\n  [STABILNOSC] GCN_gap: zawsze<0? {all(x<0 for x in gcn)} zakres[{min(gcn):+.3f},{max(gcn):+.3f}]")
    print(f"               temp_gap: zawsze>=0? {all(x>=-0.01 for x in tmp)} zakres[{min(tmp):+.3f},{max(tmp):+.3f}]")
    print(f"               struct_r: zawsze<0? {all(x<0 for x in sr)} zakres[{min(sr):+.2f},{max(sr):+.2f}]")
    if save:
        _write("robustness_grid", ["seeds", "core", "strat", "gcn_gap", "temp_gap", "struct_r", "struct_p"], rows)
    return rows


def leak_audit(graph: str = "enron", seeds: int = 3, rates=(0.0, 0.1, 0.2, 0.3),
               save: bool = True) -> List[list]:
    """Audyt wycieku rytmu: detekcja vs odsetek benign off-hours (wymaga grafu z realnym rytmem)."""
    g = load(graph); det = TemporalGNN(); rows = []
    for r in rates:
        aucs, recs = [], []
        for s in range(seeds):
            sc = build_scenario(g, CascadeStrategy(fanout=5), s, off_hours=r)
            tr, te = victim_split(sc.events, s)
            yte = np.array([e[3] for e in sc.events], float)[te]
            sco = det.fit_score(sc, tr, te, s)
            aucs.append(roc_auc_score(yte, sco)); recs.append(recall_at_fpr(yte, sco, 0.01))
        rows.append([r, round(float(np.mean(aucs)), 4), round(float(np.mean(recs)), 4)])
        print(f"  off-hours={r:.0%}: AUC={rows[-1][1]:.3f} R@1%={rows[-1][2]:.3f}", flush=True)
    if save:
        _write("leak_audit", ["off_hours", "auc", "recall_fpr1"], rows)
    return rows


def headline_ci(graph: str = "synthetic:600", seeds: int = 20, save: bool = True):
    """(C) RYGOR STAT. dla glownego twierdzenia P2: pod atakiem SKRYTYM sygnal odbiorczy
    (temporalny LUB reczny) bije wolumenowy (COMPA); temporalny >= reczny. Raportuje per-detektor
    mean + bootstrap-CI95 pod rezimem skrytym i naiwnym, oraz PAROWANY Wilcoxon + Cliff's delta dla
    kontrastow (temporalny vs reczny, sygnal-odbiorczy vs COMPA). >=20 ziaren => moc nie jest podloga."""
    from .stats import mean_ci, paired_wilcoxon, cliffs_delta, min_two_sided_wilcoxon_p
    g = load(graph)
    names = [d.name for d in get_panel()]
    regimes = {"skryty": CascadeStrategy(fanout=2, spread=8, mimicry=0.3),
               "naiwny": CascadeStrategy(fanout=8, spread=1)}
    perdet, rows = {}, []
    print(f"=== HEADLINE-CI: {seeds} ziaren, bootstrap-CI95 ({len(names)} det.) ===", flush=True)
    for rname, strat in regimes.items():
        sm = {n: [] for n in names}
        for s in range(seeds):
            ev = evaluate(build_scenario(g, strat, s), get_panel(), s)
            for n in names:
                sm[n].append(float(ev[n]["auc"]))
        perdet[rname] = sm
        print(f"  [{rname}]", flush=True)
        for n in names:
            m, lo, hi = mean_ci(sm[n])
            rows.append([rname, n, round(m, 3), round(lo, 3), round(hi, 3)])
            print(f"    {n:22s} {m:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
    # kontrasty parowane pod rezimem skrytym
    floor = min_two_sided_wilcoxon_p(seeds)
    contrasts = [("temporalny-GNN", "reczny-kontekst"), ("temporalny-GNN", "COMPA"),
                 ("reczny-kontekst", "COMPA")]
    crows = []
    print(f"\n  Kontrasty parowane (rezim skryty), podloga dwustronnego Wilcoxona n={seeds}: {floor:.2g}", flush=True)
    for x, y in contrasts:
        a, b = perdet["skryty"][x], perdet["skryty"][y]
        _, p = paired_wilcoxon(a, b)
        d, mag = cliffs_delta(a, b)
        crows.append([f"{x} vs {y}", round(float(np.mean(a)) - float(np.mean(b)), 3), round(p, 4),
                      round(d, 3), mag])
        print(f"    {x} vs {y}: dAUC={np.mean(a)-np.mean(b):+.3f}  Wilcoxon p={p:.4f}  Cliff d={d:+.2f} ({mag})", flush=True)
    if save:
        _write("headline_ci", ["rezim", "detektor", "auc_mean", "ci_lo", "ci_hi"], rows)
        _write("headline_contrasts", ["kontrast", "dAUC", "wilcoxon_p", "cliffs_d", "efekt"], crows)
    return {"perdet": perdet, "contrasts": crows}
