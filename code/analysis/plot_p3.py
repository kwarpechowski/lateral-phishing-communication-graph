"""Figury do Publikacji 3 (adwersaryjna odpornosc) -> latex_p3/figures/.
  fig_p3_pareto   — front Pareto ewazja vs zasieg
  fig_p3_adaptive — detekcja vs rozproszenie przy STALYM zasiegu (odpornosc na ewazje)
  fig_p3_panel    — panel detektorow pod strategiami atakujacego
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent.parent
RESULTS = CODE / "results"
FIG = CODE.parent / "latex_p3" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.unicode_minus": False,
                     # DejaVu Sans (bundled) ma polskie diakrytyki; fonttype 42 osadza TrueType w PDF.
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})


def fig_pareto():
    df = pd.read_csv(RESULTS / "exp_p3_pareto.csv").sort_values("infected_frac")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(df["infected_frac"] * 100, df["auc"], "o-", c="#1f77b4", lw=2)
    for _, r in df.iterrows():
        lab = "blast" if r["K"] == 999 else f"K={int(r['K'])}"
        ax.annotate(lab, (r["infected_frac"] * 100, r["auc"]), textcoords="offset points",
                    xytext=(6, -4), fontsize=8)
    ax.axhline(0.5, ls=":", c="grey", lw=0.7)
    ax.set_xlabel("zasięg ataku: % grafu zarażone"); ax.set_ylabel("wykrywalność (AUC)")
    ax.set_title("Front Pareto: ewazja kosztuje zasięg"); ax.grid(alpha=0.3)
    fig.savefig(FIG / "fig_p3_pareto.pdf"); plt.close(fig); print("[p3] fig_p3_pareto.pdf")


def fig_adaptive():
    df = pd.read_csv(RESULTS / "exp_p3_adaptive.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m, c, lab in [(0.0, "#1f77b4", "bez mimikry"), (0.5, "#2ca02c", "mimikra 50%")]:
        d = df[df.mimicry == m].sort_values("gap")
        ax.plot(d["gap"], d["auc"], "o-", c=c, label=lab)
    ax.axhline(0.5, ls=":", c="grey", lw=0.7)
    ax.set_ylim(0.45, 0.85)
    ax.set_xlabel("rozproszenie w czasie (gap)  [zasięg stały ~99%]")
    ax.set_ylabel("wykrywalność (AUC)")
    ax.set_title("Odporność na ewazję: przy stałym zasięgu\nrozproszenie nie zbija detekcji")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.savefig(FIG / "fig_p3_adaptive.pdf"); plt.close(fig); print("[p3] fig_p3_adaptive.pdf")


def fig_panel():
    df = pd.read_csv(RESULTS / "exp_p3_panel.csv")
    dets = ["1-hop", "COMPA", "GCN-statyczny", "reczny-kontekst", "temporalny-GNN"]
    labs = ["1-hop", "COMPA", "GCN\nstat.", "ręczny\nkontekst", "temporalny\nGNN"]
    strat = list(df["strategia"]); x = np.arange(len(dets)); wbar = 0.25
    cols = ["#999999", "#d62728", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for i, s in enumerate(strat):
        row = df[df.strategia == s].iloc[0]
        vals = [float(row[dd]) for dd in dets]
        ax.bar(x + (i - 1) * wbar, vals, wbar, color=cols[i],
               label=f"{s} ({row['zasieg']:.0%} zasięg)")
    ax.axhline(0.5, ls=":", c="grey", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.5); ax.set_ylim(0.45, 0.82)
    ax.set_ylabel("wykrywalność (AUC)")
    ax.set_title("Panel detektorów pod atakiem: grafowe odporne, wolumenowe/lokalne pękają przy niskim zasięgu")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_p3_panel.pdf"); plt.close(fig); print("[p3] fig_p3_panel.pdf")


if __name__ == "__main__":
    fig_pareto(); fig_adaptive(); fig_panel()
    print(f"[p3] -> {FIG}")
