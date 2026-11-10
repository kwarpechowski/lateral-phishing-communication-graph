"""Figures for Paper 2 (lateral-movement cascade + temporal GNN) -> paper/figures/.

  fig_cascade_detectors.pdf — detector hierarchy (AUC + Recall@FPR=1%) on the victim split,
                              with a shuffle control.
  fig_cascade_scale.pdf      — Recall@FPR=1% vs number of organizations (19/30/200), temporal vs hand-crafted:
                              the operational advantage holds inductively but shrinks with scale.

Labels: plain English unicode (no LaTeX escaping).
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
FIG = CODE.parent / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.unicode_minus": False,
                     # DejaVu Sans (bundled) has Polish diacritics; fonttype 42 embeds TrueType in the PDF.
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})


def _load(name):
    return pd.read_csv(RESULTS / name).set_index("model")


def fig_detectors():
    df = _load("exp_cascade.csv")
    order = ["tab_1hop", "gnn_static", "tab_ctx", "gnn_temporal", "gnn_temporal_shuf"]
    labels = ["1-hop\n(tabular)", "static\nGNN", "hand-crafted\ncontext", "temporal\nGNN",
              "temporal\n(shuffle)"]
    df = df.loc[order]
    colors = ["#999999", "#7aa6c2", "#2ca02c", "#1f77b4", "#d62728"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.7))
    a1.bar(range(len(df)), df["auc"], color=colors)
    a1.axhline(0.5, ls=":", c="grey", lw=0.8)
    for i, v in enumerate(df["auc"]):
        a1.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=9)
    a1.set_xticks(range(len(df))); a1.set_xticklabels(labels, fontsize=8)
    a1.set_ylabel("AUC"); a1.set_ylim(0.45, 0.95); a1.set_title("AUC (ranking)")
    a2.bar(range(len(df)), df["recall_fpr1"], color=colors)
    for i, v in enumerate(df["recall_fpr1"]):
        a2.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    a2.set_xticks(range(len(df))); a2.set_xticklabels(labels, fontsize=8)
    a2.set_ylabel("Sensitivity@FPR=1%"); a2.set_title("Sensitivity at FPR=1% (operational)")
    fig.suptitle("Cascade detector hierarchy: blind 1-hop → topology → hand-crafted window → learned dynamics",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cascade_detectors.pdf"); plt.close(fig)
    print("[fig] fig_cascade_detectors.pdf")


def fig_scale():
    pts = [(19, "exp_cascade_org.csv"), (30, "exp_cascade_scale240_org.csv"),
           (200, "exp_cascade_scale1600_org.csv")]
    orgs, temp, hand = [], [], []
    for o, fn in pts:
        d = _load(fn)
        orgs.append(o)
        temp.append(d.loc["gnn_temporal", "recall_fpr1"])
        hand.append(d.loc["tab_ctx", "recall_fpr1"])
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(orgs, temp, "o-", c="#1f77b4", label="temporal GNN")
    ax.plot(orgs, hand, "s--", c="#2ca02c", label="hand-crafted context")
    ax.fill_between(orgs, hand, temp, alpha=0.12, color="#1f77b4")
    for o, t, h in zip(orgs, temp, hand):
        ha = "left" if o == orgs[0] else ("right" if o == orgs[-1] else "center")
        ax.text(o, t + 0.004, f"{t:.3f}", ha=ha, fontsize=8)
        ax.text(o, h + 0.005, f"{h:.3f}", ha=ha, fontsize=8, color="#2ca02c")  # above the point, to avoid colliding with the X axis
    ax.set_xscale("log")
    ax.set_xticks(orgs); ax.set_xticklabels([str(o) for o in orgs])
    ax.set_xlabel("number of organizations (inductive split)")
    ax.set_ylabel("Sensitivity@FPR=1% (unseen organizations)")
    ax.set_title("The operational advantage holds inductively,\nbut shrinks with scale")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.savefig(FIG / "fig_cascade_scale.pdf"); plt.close(fig)
    print("[fig] fig_cascade_scale.pdf")


def fig_enron():
    df = _load("exp_cascade_enron.csv")
    order = ["tab_1hop", "tab_ctx", "gnn_temporal", "gnn_temporal_shuf"]
    labels = ["1-hop", "hand-crafted\ncontext", "temporal\nGNN", "temporal\n(shuffle)"]
    df = df.loc[order]
    colors = ["#999999", "#2ca02c", "#1f77b4", "#d62728"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.bar(range(len(df)), df["auc"], color=colors)
    a1.axhline(0.5, ls=":", c="grey", lw=0.8)
    for i, v in enumerate(df["auc"]):
        a1.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=9)
    a1.set_xticks(range(len(df))); a1.set_xticklabels(labels, fontsize=8)
    a1.set_ylabel("AUC"); a1.set_ylim(0.45, 0.92); a1.set_title("AUC")
    a2.bar(range(len(df)), df["recall_fpr1"], color=colors)
    for i, v in enumerate(df["recall_fpr1"]):
        a2.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9)
    a2.set_xticks(range(len(df))); a2.set_xticklabels(labels, fontsize=8)
    a2.set_ylabel("Sensitivity@FPR=1%"); a2.set_title("Sensitivity at FPR=1%")
    fig.suptitle("Anchor: REAL Enron contact graph (1990 nodes, 8 seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cascade_enron.pdf"); plt.close(fig)
    print("[fig] fig_cascade_enron.pdf")


def fig_cascade_graph():
    """Visualization of cascade propagation on the contact graph (ring layout by hop)."""
    import sys
    sys.path.insert(0, str(CODE))
    from graph.build_layers import build_layers
    base = build_layers()
    nbr = {}
    for a, b in base["contact"]:
        nbr.setdefault(a, set()).add(b); nbr.setdefault(b, set()).add(a)
    rng = np.random.default_rng(3)
    seed0 = max(nbr, key=lambda x: len(nbr[x]))                      # hubs = interesting cascade
    parent = {seed0: None}; rings = [[seed0]]; cur = [seed0]; seen = {seed0}
    edges = []
    for _hop in range(3):
        nxt = []
        for u in cur:
            for v in sorted(nbr[u]):
                if v in seen:
                    continue
                if rng.random() < 0.55:
                    seen.add(v); parent[v] = u; nxt.append(v); edges.append((u, v))
        if not nxt:
            break
        rings.append(nxt); cur = nxt
    pos = {seed0: (0.0, 0.0)}
    for r, ring in enumerate(rings):
        if r == 0:
            continue
        for i, node in enumerate(ring):
            ang = 2 * np.pi * i / len(ring) + 0.5 * r
            pos[node] = (r * np.cos(ang), r * np.sin(ang))
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for u, v in edges:
        if u in pos and v in pos:
            ax.annotate("", xy=pos[v], xytext=pos[u],
                        arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=1.1, alpha=0.7))
    ringcol = ["#b30000", "#1f77b4", "#5599dd", "#aaccee"]
    for r, ring in enumerate(rings):
        xs = [pos[n][0] for n in ring]; ys = [pos[n][1] for n in ring]
        ax.scatter(xs, ys, s=(180 if r == 0 else 90), c=ringcol[min(r, 3)],
                   edgecolors="black", linewidths=0.6, zorder=3,
                   label=("carrier ($s_0$)" if r == 0 else f"hop {r}"))
    ax.scatter([], [], c="white", label=f"{len(seen)} accounts total")
    ax.set_title("Cascade propagation on the real contact graph\n(red = malicious sends)")
    ax.legend(fontsize=8, loc="upper left"); ax.axis("off"); ax.set_aspect("equal")
    fig.savefig(FIG / "fig_cascade_graph.pdf", bbox_inches="tight"); plt.close(fig)
    print("[fig] fig_cascade_graph.pdf")


def fig_stealth():
    df = pd.read_csv(RESULTS / "exp_cascade_stealth.csv")
    order = [1, 2, 3, 5, 999]; xlab = ["K=1", "K=2", "K=3", "K=5", "blast"]
    xs = list(range(len(order)))
    def series(m, col):
        return [float(df[(df.K == k) & (df.model == m)][col].iloc[0]) for k in order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for m, c, lab, st in [("compa", "#d62728", "volumetric (per-account)", "s--"),
                          ("tab_ctx", "#2ca02c", "hand-crafted context", "^:"),
                          ("gnn_temporal", "#1f77b4", "temporal GNN", "o-")]:
        a1.plot(xs, series(m, "auc"), st, c=c, label=lab)
        a2.plot(xs, series(m, "recall_fpr1"), st, c=c, label=lab)
    for ax, t in [(a1, "AUC"), (a2, "Sensitivity@FPR=1%")]:
        ax.axhline(0.5 if t == "AUC" else 0, ls=":", c="grey", lw=0.7)
        ax.set_xticks(xs); ax.set_xticklabels(xlab, fontsize=9)
        ax.set_xlabel("← stealthier      naive blast →", fontsize=8)
        ax.set_title(t)
    a1.set_ylabel("AUC"); a2.set_ylabel("Sensitivity@FPR=1%"); a1.legend(fontsize=8, loc="lower right")
    fig.suptitle("Stealthy attacker: the volumetric baseline collapses, the temporal GNN maintains the advantage", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cascade_stealth.pdf"); plt.close(fig)
    print("[fig] fig_cascade_stealth.pdf")


def fig_realgraphs():
    df = pd.read_csv(RESULTS / "exp_cascade_realgraphs.csv")
    dsets = ["email-Eu-core", "CollegeMsg"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, ds in zip(axes, dsets):
        d = df[df.dataset == ds]
        regimes = ["blast", "stealthy"]
        def val(m, r):
            return float(d[(d.model == m) & (d.regime == r)]["auc"].iloc[0])
        x = np.arange(len(regimes)); w = 0.35
        ax.bar(x - w / 2, [val("compa", r) for r in regimes], w, color="#d62728", label="volumetric")
        ax.bar(x + w / 2, [val("gnn_temporal", r) for r in regimes], w, color="#1f77b4", label="temporal GNN")
        for i, r in enumerate(regimes):
            ax.text(i - w / 2, val("compa", r) + 0.01, f"{val('compa', r):.2f}", ha="center", fontsize=8)
            ax.text(i + w / 2, val("gnn_temporal", r) + 0.01, f"{val('gnn_temporal', r):.2f}", ha="center", fontsize=8)
        ax.axhline(0.5, ls=":", c="grey", lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels(["naive\nblast", "stealthy"], fontsize=9)
        ax.set_title(ds, fontsize=10); ax.set_ylim(0.45, 1.0)
    axes[0].set_ylabel("AUC"); axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Real topologies: volumetric wins the blast, collapses under the stealthy attacker; temporal GNN maintains the advantage",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cascade_realgraphs.pdf"); plt.close(fig)
    print("[fig] fig_cascade_realgraphs.pdf")


def fig_fpr():
    df = pd.read_csv(RESULTS / "exp_cascade_fpr.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    sty = {"compa": ("s--", "#d62728", "volumetric"), "tab_ctx": ("^:", "#2ca02c", "hand-crafted context"),
           "gnn_temporal": ("o-", "#1f77b4", "temporal GNN")}
    for ax, reg in zip(axes, ["blast", "stealthy"]):
        d = df[df.regime == reg]
        for m, (st, c, lab) in sty.items():
            dd = d[d.model == m].sort_values("fpr")
            ax.plot(dd["fpr"], dd["recall"], st, c=c, label=lab)
        ax.set_xscale("log"); ax.set_xlabel("FPR (log scale)")
        ax.set_title({"blast": "naive blast", "stealthy": "stealthy attack"}.get(reg, reg), fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Sensitivity"); axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Sensitivity–FPR curve (10 seeds): at low FPR all methods are weak; "
                 "volumetric wins the blast, collapses under the stealthy attacker", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cascade_fpr.pdf"); plt.close(fig)
    print("[fig] fig_cascade_fpr.pdf")


def fig_p3_pareto():
    df = pd.read_csv(RESULTS / "exp_p3_pareto.csv").sort_values("reach_events")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = df["infected_frac"] * 100; ydet = df["auc"]
    ax.plot(x, ydet, "o-", c="#1f77b4", lw=2)
    for _, r in df.iterrows():
        lab = "blast" if r["K"] == 999 else f"K={int(r['K'])}"
        ax.annotate(lab, (r["infected_frac"] * 100, r["auc"]),
                    textcoords="offset points", xytext=(6, -4), fontsize=8)
    ax.axhline(0.5, ls=":", c="grey", lw=0.7)
    ax.set_xlabel("attack reach: % of graph infected")
    ax.set_ylabel("detectability (temporal GNN AUC)")
    ax.set_title("Pareto front: evasion costs reach\n(the attacker cannot have low detection and high reach at once)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_p3_pareto.pdf"); plt.close(fig)
    print("[fig] fig_p3_pareto.pdf")


if __name__ == "__main__":
    fig_detectors()
    fig_scale()
    fig_enron()
    fig_cascade_graph()
    fig_stealth()
    fig_realgraphs()
    fig_fpr()
    try:
        fig_p3_pareto()
    except FileNotFoundError:
        print("[fig] (skipping p3_pareto — missing csv)")
    print(f"[fig] -> {FIG}")
