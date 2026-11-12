#!/usr/bin/env python3
"""Generuje figury P3 z wynikow cb_*.csv -> latex_p3/figures/ (PDF).

Uruchom z katalogu code/:  ../../venv/Scripts/python cascadebench/plot_cb.py
Czyta:  results/cb_*.csv
Pisze:  ../latex_p3/figures/*.pdf
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# DejaVu Sans (bundled z matplotlib) ma polskie diakrytyki; fonttype 42 osadza TrueType w PDF.
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.unicode_minus": False})

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGS = HERE.parents[1] / "latex_p3" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# kolejnosc i kolory detektorow (spojne miedzy figurami)
DETS = ["1-hop", "COMPA", "hopper", "anomaly-forest", "GCN-statyczny",
        "reczny-kontekst", "temporalny-GNN", "temporalny-GNN-uwaga"]
COLOR = {
    "1-hop": "#9e9e9e", "COMPA": "#d62728", "hopper": "#bcbd22", "anomaly-forest": "#8c564b",
    "GCN-statyczny": "#1f77b4", "reczny-kontekst": "#2ca02c",
    "temporalny-GNN": "#ff7f0e", "temporalny-GNN-uwaga": "#9467bd",
}
MARK = {
    "1-hop": "o", "COMPA": "s", "hopper": "X", "anomaly-forest": "^", "GCN-statyczny": "D",
    "reczny-kontekst": "v", "temporalny-GNN": "P", "temporalny-GNN-uwaga": "*",
}
# Nazwy wyswietlane: nie atrybuujemy waskich reimplementacji pelnym systemom (recenzja P3 #18).
DISP = {"COMPA": "wolumenowy", "hopper": "ścieżkowy"}
def _disp(name): return DISP.get(name, name)


def _read(name):
    with open(RESULTS / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_robustness():
    rows = _read("cb_robustness_curves.csv")
    reach = [float(r["zasieg"]) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for d in DETS:
        if d not in rows[0]:
            continue
        ys = [float(r[d]) for r in rows]
        ax.plot(reach, ys, marker=MARK[d], color=COLOR[d], label=_disp(d),
                linewidth=1.8, markersize=7)
    ax.set_xlabel("Zasięg ataku [% węzłów]  (fan-out 2→8)")
    ax.set_ylabel("Detekcja (AUC)")
    ax.set_ylim(0.50, 0.88)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower center")
    ax.set_title("Odporność detektorów względem zasięgu atakującego")
    fig.tight_layout()
    out = FIGS / "robustness_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_panel():
    rows = _read("cb_panel.csv")
    strat = [r["strategia"] for r in rows]
    dets = [c for c in rows[0] if c not in ("strategia", "zasieg")]
    x = range(len(strat))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    n = len(dets)
    w = 0.8 / n
    for i, d in enumerate(dets):
        ys = [float(r[d]) for r in rows]
        xs = [xi + (i - n / 2) * w + w / 2 for xi in x]
        ax.bar(xs, ys, width=w, color=COLOR.get(d, "#555"), label=_disp(d))
    ax.set_xticks(list(x))
    ax.set_xticklabels(["naiwna\n(zasięg 94%)", "rozproszona\n(zasięg 94%)", "niski zasięg\n(54%)"])
    ax.set_ylabel("Detekcja (AUC)")
    ax.set_ylim(0.4, 0.9)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_title("Panel detektorów względem strategii atakującego")
    fig.tight_layout()
    out = FIGS / "panel.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_validate_real():
    rows = _read("cb_validate_real.csv")
    graphs = [r["graph"] for r in rows]
    dets = [c for c in rows[0] if c != "graph"]
    x = range(len(graphs))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    n = len(dets)
    w = 0.8 / n
    for i, d in enumerate(dets):
        ys = [float(r[d]) for r in rows]
        xs = [xi + (i - n / 2) * w + w / 2 for xi in x]
        ax.bar(xs, ys, width=w, color=COLOR.get(d, "#555"), label=_disp(d))
    ax.set_xticks(list(x))
    ax.set_xticklabels([g.replace("synthetic:600", "syntetyk") for g in graphs])
    ax.set_ylabel("Detekcja (AUC)")
    ax.set_ylim(0.4, 0.9)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_title("Syntetyk vs realne topologie: rankingi nie transferują")
    fig.tight_layout()
    out = FIGS / "validate_real.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_predictivity():
    """Scatter: dystans strukturalny vs transfer rang detektorow (rdzen sondy predyktywnosci)."""
    fp = RESULTS / "cb_predictivity_transfer.csv"
    if not fp.exists():
        print("skip predictivity (brak CSV)")
        return
    rows = _read("cb_predictivity_transfer.csv")
    xs = [float(r["struct_dist"]) for r in rows]
    ys = [float(r["rank_transfer"]) for r in rows]
    labels = [r["graph"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(xs, ys, s=70, color="#9467bd", zorder=3, edgecolor="k", linewidth=0.5)
    for x, y, lb in zip(xs, ys, labels):
        ax.annotate(lb, (x, y), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    # linia trendu
    if len(xs) >= 2:
        import numpy as np
        a, b = np.polyfit(xs, ys, 1)
        xx = [min(xs), max(xs)]
        ax.plot(xx, [a * x + b for x in xx], "--", color="#d62728", lw=1.5,
                label=f"trend (nachylenie {a:.2f})")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("Dystans strukturalny od grafu treningowego\n(skos stopni + klasteryzacja + gęstość)")
    ax.set_ylabel("Transfer rang detektorów (Spearman ρ)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Sonda predyktywności: transfer rang maleje z dystansem strukturalnym")
    fig.tight_layout()
    out = FIGS / "predictivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_predictivity_gap():
    """Bar: luka transferu syntetyk->real per detektor (GLOWNY wynik sondy).
    Ujemna = detektor zawyzany przez syntetyk; dodatnia/zero = transferuje wiernie."""
    fp = RESULTS / "cb_predictivity_gap.csv"
    if not fp.exists():
        print("skip predictivity_gap (brak CSV)")
        return
    rows = _read("cb_predictivity_gap.csv")
    labels = [r["detektor"] for r in rows]
    gaps = [float(r["gap"]) for r in rows]
    cis = [float(r.get("gap_ci95", 0) or 0) for r in rows]
    cols = [COLOR.get(l, "#555") for l in labels]
    order = sorted(range(len(gaps)), key=lambda i: gaps[i])
    labels = [labels[i] for i in order]; gaps = [gaps[i] for i in order]
    cis = [cis[i] for i in order]; cols = [cols[i] for i in order]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh(range(len(gaps)), gaps, xerr=cis, color=cols, edgecolor="k", linewidth=0.5,
            error_kw=dict(ecolor="#333", capsize=3, lw=1))
    ax.set_yticks(range(len(gaps))); ax.set_yticklabels([_disp(l) for l in labels], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Luka transferu AUC (real − syntetyk),  wąsy = 95% CI")
    ax.set_title("Luka transferu syntetyk→real per detektor (20 ziaren)")
    fig.tight_layout()
    out = FIGS / "predictivity_gap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_clustering():
    """Scatter klasteryzacja grafu vs AUC detektora (M2): GCN/wolumenowy rosną z klasteryzacją (zawyżenie),
    temporalne płaskie (odporne). Mechanizm headline'u."""
    fp = RESULTS / "cb_predictivity.csv"
    if not fp.exists():
        print("skip clustering (brak CSV)")
        return
    rows = _read("cb_predictivity.csv")
    clust = [float(r["clustering"]) for r in rows]
    show = ["GCN-statyczny", "COMPA", "temporalny-GNN", "temporalny-GNN-uwaga"]
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for d in show:
        if d not in rows[0]:
            continue
        ys = [float(r[d]) for r in rows]
        ax.scatter(clust, ys, s=45, color=COLOR.get(d, "#555"), marker=MARK.get(d, "o"),
                   edgecolor="k", linewidth=0.4, zorder=3, label=_disp(d))
        if len(set(clust)) >= 2:
            a, b = np.polyfit(clust, ys, 1)
            xx = np.array([min(clust), max(clust)])
            ax.plot(xx, a * xx + b, "--", color=COLOR.get(d, "#555"), lw=1.2, alpha=0.7)
    ax.set_xlabel("Klasteryzacja grafu treningowego")
    ax.set_ylabel("AUC detektora")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Zawyżenie ∝ klasteryzacji: GCN/wolumenowy rosną, temporalne płaskie")
    fig.tight_layout()
    out = FIGS / "clustering.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_fabrication():
    """(D) Detekcja vs frakcja fabrykacji krawedzi per detektor."""
    fp = RESULTS / "cb_fabrication.csv"
    if not fp.exists():
        print("skip fabrication (brak CSV)"); return
    rows = _read("cb_fabrication.csv")
    xs = [float(r["fabrication"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for d in DETS:
        if d not in rows[0]:
            continue
        ax.plot(xs, [float(r[d]) for r in rows], marker=MARK[d], color=COLOR[d], label=_disp(d), lw=1.6)
    ax.set_xlabel("Frakcja fabrykacji krawędzi (kontakty spoofowane OSINT)")
    ax.set_ylabel("Detekcja (AUC)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2)
    ax.set_title("Atak fabrykujący krawędzie vs panel detektorów")
    fig.tight_layout()
    out = FIGS / "fabrication.pdf"; fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def fig_mechanism():
    """(C) AUC vs modularność Q, rozdzielone na rodziny SBM (niska klast.) i kliki (wysoka klast.)."""
    fp = RESULTS / "cb_mechanism.csv"
    if not fp.exists():
        print("skip mechanism (brak CSV)"); return
    rows = _read("cb_mechanism.csv")
    show = ["GCN-statyczny", "temporalny-GNN", "COMPA"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for fam, mk, ls in (("SBM", "o", "-"), ("clique", "s", "--")):
        fr = [r for r in rows if r["family"] == fam]
        if not fr:
            continue
        Q = [float(r["modularity_Q"]) for r in fr]
        for d in show:
            if d not in fr[0]:
                continue
            ax.plot(Q, [float(r[d]) for r in fr], marker=mk, ls=ls, color=COLOR.get(d, "#555"),
                    lw=1.5, label=f"{d} [{fam}]")
    ax.set_xlabel("Modularność Q  (SBM: niska klasteryzacja; kliki: wysoka)")
    ax.set_ylabel("Detekcja (AUC)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=6, ncol=2)
    ax.set_title("Mechanizm: GCN reaguje na strukturę klik, nie na samą modularność")
    fig.tight_layout()
    out = FIGS / "mechanism.pdf"; fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    fig_robustness()
    fig_panel()
    fig_validate_real()
    fig_predictivity()
    fig_predictivity_gap()
    fig_clustering()
    fig_fabrication()
    fig_mechanism()
    print("DONE")
