"""Brakujacy generator tabel cb_hopper_stealth (Tab. baselines) i cb_labelnoise (Tab. S4).

Odtworzony po audycie reprodukcyjnosc: te CSV-y istnialy w results/, ale skrypt je
generujacy nie byl w pakiecie. Protokol (dokumentowany w artykule A, S6):
  * harness cascadebench (graf synthetic:600, panel: COMPA/hopper/reczny-kontekst/temporalny-GNN)
  * STEALTH SWEEP: K in {1,2,3,5}; CascadeStrategy(fanout=K, spread=8, mimicry=0.3)
    (rezym skryty jak w headline_ci), benign matched 3:1, off_hours=0.15, split po ofiarach
  * LABEL NOISE: rezym skryty K=3 (punkt srodkowy sweepu); szum etykiet w zbiorze TRENINGOWYM
    (flip frakcji etykiet), test bez szumu; poziomy {0, 5, 10, 20}%
  * 20 ziaren w obu eksperymentach (zgodnie z deklaracja "stealth/baselines/label noise = 20 seedow")

Wyjscie: results/cb_hopper_stealth.csv, results/cb_labelnoise.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results"

from cascadebench.attack import CascadeStrategy, build_scenario  # noqa: E402
from cascadebench.detect import COMPA, Hopper, HandContext, TemporalGNN  # noqa: E402
from cascadebench.features import hopper as hopper_feat  # noqa: E402


class HopperNoPrecursor(Hopper):
    """Ablacja po recenzji zewnetrznej: cecha precursor jest identyczna z freshness
    kontekstu recznego, wiec wynik hoppera przy K=1 moze byc wnoszony przez nia.
    Wersja bez precursor (i bez pochodnego hop_sig = precursor*unexplained)."""

    name = "hopper-bez-precursor"

    def _X(self, sc, seed):
        import numpy as _np

        X = _np.array(hopper_feat(sc.graph, sc.events, seed), dtype=_np.float32).copy()
        X[:, 0] = 0.0
        X[:, 4] = 0.0
        return X


from cascadebench.evaluate import evaluate, victim_split, recall_at_fpr  # noqa: E402
from cascadebench.graph import load  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

SEEDS = 20
K_GRID = [1, 2, 3, 5]
NOISE_GRID = [0.0, 0.05, 0.1, 0.2]
OFF_HOURS = 0.15
PANEL = [COMPA, Hopper, HopperNoPrecursor, HandContext, TemporalGNN]
NAMES = [d().name for d in PANEL]


def _strat(K, timing="det"):
    return CascadeStrategy(fanout=K, spread=8, mimicry=0.3, timing=timing)


def _timing_stats(events, t_buckets):
    """R4-A1: KS odstepow atak vs tlo + sonda czysto czasowa (per seed)."""
    from scipy.stats import ks_2samp

    from exp_injection_probe import _gaps, _probe_auc

    a_g, _ = _gaps(events, 1)
    b_g, _ = _gaps(events, 0)
    if len(a_g) >= 5 and len(b_g) >= 5:
        ks, p = ks_2samp(a_g, b_g)
        ks = float(ks)
    else:
        ks, p = float("nan"), float("nan")
    probe = _probe_auc(list(events), t_buckets, 123)
    return ks, float(p), probe


def stealth_sweep():
    g = load("synthetic:600")
    rows = []
    perseed = []
    timing_rows = []
    for timing in ("det", "emp"):  # R4-A1: kontrola sygnatury iniekcji
        configs = [(f"K={K}", _strat(K, timing)) for K in K_GRID]
        configs.append(
            ("blast", CascadeStrategy(fanout=8, spread=1, timing=timing))
        )  # sanity ceiling dla bloku (a)
        for level, strat in configs:
            acc = {n: {"auc": [], "r1": []} for n in NAMES}
            infe = []
            ks_l, pr_l = [], []
            for s in range(SEEDS):
                sc = build_scenario(g, strat, s, off_hours=OFF_HOURS)
                infe.append(sc.infected_frac)
                ev = evaluate(sc, [d() for d in PANEL], s)
                for n in NAMES:
                    acc[n]["auc"].append(ev[n]["auc"])
                    acc[n]["r1"].append(ev[n]["recall@0.01"])
                    perseed.append(
                        [
                            timing,
                            level,
                            s,
                            n,
                            round(ev[n]["auc"], 4),
                            round(ev[n]["recall@0.01"], 4),
                        ]
                    )
                ks, _kp, probe = _timing_stats(sc.events, g.n_buckets)
                ks_l.append(ks)
                pr_l.append(probe)
            rows.append(
                [timing, level, round(float(np.mean(infe)), 3)]
                + [round(float(np.mean(acc[n]["auc"])), 4) for n in NAMES]
                + [round(float(np.mean(acc[n]["r1"])), 4) for n in NAMES]
            )
            timing_rows.append(
                [
                    timing,
                    level,
                    round(float(np.nanmean(ks_l)), 4),
                    round(float(np.mean(pr_l)), 4),
                ]
            )
            print(
                f"  [{timing}] {level} (zasieg {np.mean(infe):.2f}): "
                + " ".join(f"{n}={np.mean(acc[n]['auc']):.3f}" for n in NAMES)
                + f" | KS={np.nanmean(ks_l):.2f} probe={np.mean(pr_l):.2f}",
                flush=True,
            )
    out = RESULTS / "cb_hopper_stealth.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["timing", "poziom", "zasieg"]
            + [f"auc_{n}" for n in NAMES]
            + [f"r1_{n}" for n in NAMES]
        )
        w.writerows(rows)
    print(f"[cb-stealth] -> {out}")
    out_t = RESULTS / "cb_hopper_stealth_timing.csv"
    with out_t.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timing", "poziom", "ks_mean", "probe_auc"])
        w.writerows(timing_rows)
    print(f"[cb-stealth-timing] -> {out_t}")
    out_ps = RESULTS / "cb_hopper_stealth_perseed.csv"
    with out_ps.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timing", "poziom", "seed", "detektor", "auc", "r1"])
        w.writerows(perseed)
    print(f"[cb-stealth-perseed] -> {out_ps}")


def label_noise():
    from cascadebench.attack import Scenario

    g = load("synthetic:600")
    rows = []
    perseed = []
    for q in NOISE_GRID:
        acc = {n: [] for n in NAMES}
        for s in range(SEEDS):
            sc = build_scenario(g, _strat(3), s, off_hours=OFF_HOURS)
            tr, te = victim_split(sc.events, s)
            y = np.array([e[3] for e in sc.events], dtype=np.float32)
            if (
                q > 0
            ):  # szum etykiet TYLKO w zbiorze treningowym (flip etykiet zdarzen train)
                rng = np.random.default_rng(1000 + s)
                flip = rng.random(len(y)) < q
                idx_tr = np.where(tr & flip)[0]
                ev_noisy = list(sc.events)
                for i in idx_tr:
                    s_, v_, b_, _l = ev_noisy[i]
                    ev_noisy[i] = (s_, v_, b_, 1 - _l)
                sc = Scenario(
                    sc.graph, ev_noisy, sc.n_attacks, sc.infected_frac, sc.strategy
                )
            yte = y[te]
            for det in [d() for d in PANEL]:
                sc_score = det.fit_score(sc, tr, te, s)
                a_ = float(roc_auc_score(yte, sc_score))
                r_ = recall_at_fpr(yte, sc_score, 0.01)
                acc[det.name].append((a_, r_))
                perseed.append([q, s, det.name, round(a_, 4), round(r_, 4)])
        rows.append(
            [q] + [round(float(np.mean([a for a, _ in acc[n]])), 4) for n in NAMES]
        )
        print(
            f"  noise={q:.0%}: "
            + " ".join(f"{n}={np.mean([a for a, _ in acc[n]]):.3f}" for n in NAMES),
            flush=True,
        )
        print(
            f"  noise={q:.0%}: "
            + " ".join(f"{n}={np.mean([a for a, _ in acc[n]]):.3f}" for n in NAMES),
            flush=True,
        )
    out = RESULTS / "cb_labelnoise.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["szum"] + NAMES)
        w.writerows(rows)
    print(f"[cb-labelnoise] -> {out}")
    out_ps = RESULTS / "cb_labelnoise_perseed.csv"
    with out_ps.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["szum", "seed", "detektor", "auc", "r1"])
        w.writerows(perseed)
    print(f"[cb-labelnoise-perseed] -> {out_ps}")


def surgical_sweep():
    """Blok (b) Tab. baselines: atak chirurgiczny BEZ propagacji (p_inf=0), sweep po fan-out.
    Generuje cb_surgical_fanout.csv (poprzedni cb_surgical.csv nie mial generatora w pakiecie)."""
    g = load("synthetic:600")
    rows = []
    perseed = []
    for F in (1, 2, 3, 5):
        strat = CascadeStrategy(fanout=F, p_infect=0.0, spread=8)
        acc = {n: {"auc": []} for n in NAMES}
        infe = []
        for s in range(SEEDS):
            sc = build_scenario(g, strat, s, off_hours=OFF_HOURS)
            infe.append(sc.infected_frac)
            ev = evaluate(sc, [d() for d in PANEL], s)
            for n in NAMES:
                acc[n]["auc"].append(ev[n]["auc"])
                perseed.append([F, s, n, round(ev[n]["auc"], 4)])
        rows.append(
            [F, round(float(np.mean(infe)), 3)]
            + [round(float(np.mean(acc[n]["auc"])), 4) for n in NAMES]
        )
        print(
            f"  fanout={F} (zasieg {np.mean(infe):.2f}): "
            + " ".join(f"{n}={np.mean(acc[n]['auc']):.3f}" for n in NAMES),
            flush=True,
        )
    out = RESULTS / "cb_surgical_fanout.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fanout", "zasieg"] + [f"auc_{n}" for n in NAMES])
        w.writerows(rows)
    print(f"[cb-surgical] -> {out}")
    out_ps = RESULTS / "cb_surgical_fanout_perseed.csv"
    with out_ps.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fanout", "seed", "detektor", "auc"])
        w.writerows(perseed)
    print(f"[cb-surgical-perseed] -> {out_ps}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    mode = sys.argv[1] if len(sys.argv) > 1 else "stealth"
    if mode == "labelnoise":
        label_noise()
    elif mode == "surgical":
        surgical_sweep()
    else:
        stealth_sweep()
