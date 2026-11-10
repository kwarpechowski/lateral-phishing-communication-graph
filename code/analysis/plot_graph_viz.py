"""Multiplex graph visualization: all layers and connections (-> paper/figures/).

Figures:
  fig_multiplex_3d.pdf — 3D stack of layers (each layer = a plane; identical node positions;
                         intra-layer edges + dashed inter-layer connectors).
  fig_graph_overlay.pdf— 2D, all layers overlaid; edges colored by layer,
                         nodes colored by organization.

Node positions: spring_layout on the contact graph (seed=42), shared across all layers,
so that differences in connectivity between layers are visible. networkx + matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CODE = Path(__file__).resolve().parent.parent
RESULTS = CODE / "results"
FIG = CODE.parent / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "savefig.bbox": "tight", "axes.unicode_minus": False,
                     # DejaVu Sans has Polish diacritics; fonttype 42 embeds TrueType in the PDF.
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})

# Layers to visualize: contacts + 4 real OSINT layers (from fields c3/c4/c5) + time (6 layers).
LAYER_SPEC = [
    ("contact", "contacts", "#1f77b4"),
    ("osint_conference", "OSINT: conferences", "#2ca02c"),
    ("osint_certification", "OSINT: certificates", "#9467bd"),
    ("osint_skill", "OSINT: skills", "#17becf"),
    ("osint_platform", "OSINT: platforms", "#8c564b"),
    ("temporal_coactivity", "time", "#ff7f0e"),
]


def _load():
    self_df = pd.read_csv(RESULTS / "twin_self.csv")
    org = dict(zip(self_df["twin_id"], self_df["org"]))
    nodes = sorted(org)
    layers = {}
    for name, _lab, _c in LAYER_SPEC:                      # each layer from its own file
        df = pd.read_csv(RESULTS / "layers" / f"{name}.csv")
        layers[name] = list(zip(df["twin_a"], df["twin_b"]))
    return nodes, org, layers


def _layout(nodes, layers):
    G = nx.Graph(); G.add_nodes_from(nodes); G.add_edges_from(layers["contact"])
    return nx.spring_layout(G, seed=42, k=0.6, iterations=80)


def fig_3d(nodes, org, layers, pos):
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    zsep = 1.0
    sample = nodes[::6]                                    # subset of nodes for the connectors
    for li, (name, lab, col) in enumerate(LAYER_SPEC):
        z = li * zsep
        xs = [pos[n][0] for n in nodes]; ys = [pos[n][1] for n in nodes]
        ax.scatter(xs, ys, [z] * len(nodes), s=6, c=col, alpha=0.7, depthshade=False)
        for a, b in layers[name]:
            if a in pos and b in pos:
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], [z, z],
                        c=col, lw=0.25, alpha=0.35)
    zlevels = [i * zsep for i in range(len(LAYER_SPEC))]
    for n in sample:                                       # dashed inter-layer connectors
        x, y = pos[n]
        ax.plot([x] * len(zlevels), [y] * len(zlevels), zlevels, c="grey", lw=0.3, ls=":", alpha=0.5)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-60)
    ax.set_title("Multiplex graph: 150 nodes, layers stacked")
    # Named layers as a LEGEND (instead of floating 3D text that bbox='tight' clipped).
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color=col, lw=2, markersize=6, label=lab)
               for _n, lab, col in LAYER_SPEC]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.92, 0.5),
              fontsize=9, framealpha=0.9, title="Layers (bottom->top)")
    fig.savefig(FIG / "fig_multiplex_3d.pdf", dpi=200); plt.close(fig)
    print("[viz] fig_multiplex_3d.pdf")


def fig_overlay(nodes, org, layers, pos):
    orgs = sorted(set(org.values()))
    cmap = plt.get_cmap("tab20")
    ocol = {o: cmap(i % 20) for i, o in enumerate(orgs)}
    fig, ax = plt.subplots(figsize=(8, 7))
    # edges by layer (order: time, OSINT, contacts on top)
    for name, lab, col in reversed(LAYER_SPEC):
        for a, b in layers[name]:
            if a in pos and b in pos:
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                        c=col, lw=0.3, alpha=0.30)
    for n in nodes:
        ax.scatter(*pos[n], s=22, c=[ocol[org[n]]], edgecolors="white", linewidths=0.3, zorder=3)
    # Layer legend intentionally OMITTED — shared with the 3D figure (same layer colors),
    # which sits just above this one in the composite figure; avoids duplicated top/bottom labels.
    ax.set_axis_off()
    ax.set_title("All layers overlaid (edges by layer, nodes by organization)")
    fig.savefig(FIG / "fig_graph_overlay.pdf", dpi=200); plt.close(fig)
    print("[viz] fig_graph_overlay.pdf")


def main():
    nodes, org, layers = _load()
    pos = _layout(nodes, layers)
    for name, _l, _c in LAYER_SPEC:
        print(f"  layer {name}: {len(layers[name])} edges")
    fig_3d(nodes, org, layers, pos)
    fig_overlay(nodes, org, layers, pos)
    print(f"[viz] -> {FIG}")


if __name__ == "__main__":
    main()
