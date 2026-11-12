"""Generates paper figures from experiment results (matplotlib -> paper/figures/).

Figures:
  fig_ablation.pdf   — layer ablation (contribution of each layer; H8d)
  fig_pcross.pdf     — cross-org overlap sweep (H9a)
  fig_fabrication.pdf— adversarial fabrication crossover (H9d)
  fig_overlap.pdf    — Jaccard layer-coverage matrix (correlation caveat)

Labels: plain English unicode (DejaVu). No LaTeX-style escapes in matplotlib text.
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
plt.rcParams.update(
    {
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,
        # DejaVu Sans (bundled with matplotlib) covers all required glyphs;
        # fonttype 42 = TrueType embedding so glyphs are not lost in the PDF.
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def fig_ablation():
    # Reads labels DYNAMICALLY from CSV (real OSINT categories from twin fields c3-c8),
    # in the cumulative order written by exp_extended.ablation().
    df = pd.read_csv(RESULTS / "exp_ablation.csv")
    _pl_to_en = {
        "kontakty": "contacts",
        "+konferencje": "+conferences",
        "+certyfikaty": "+certificates",
        "+umiejetnosci": "+skills",
        "+rutyna": "+routine",
        "+wydarzenia": "+events",
        "+pretexty": "+pretexts",
        "+platformy": "+platforms",
        "+zainteresowania": "+interests",
        "+czasowa (pelny)": "+temporal\n(full)",
    }
    labels = [_pl_to_en.get(c, c) for c in df["config"]]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    colors = ["#888888"] * (len(df) - 1) + [
        "#1f77b4"
    ]  # last bar (full +temporal) highlighted
    ax.bar(range(len(df)), df["auc"], color=colors)
    ax.axhline(0.5, ls=":", c="grey", lw=0.8)
    for i, a in enumerate(df["auc"]):
        ax.text(i, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, 0.92)
    ax.set_title("Layer ablation (real OSINT categories): temporal layer dominates")
    fig.tight_layout()
    fig.savefig(FIG / "fig_ablation.pdf")
    plt.close(fig)
    print("[fig] fig_ablation.pdf")


def fig_pcross():
    df = pd.read_csv(RESULTS / "exp_sweep_pcross.csv")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(
        df["p_cross"], df["static_auc"], "o-", c="#1f77b4", label="static multiplex"
    )
    ax.axhline(0.5, ls=":", c="grey", lw=0.8, label="random")
    ax.set_xlabel("p_cross (cross-organization coverage of OSINT layers)")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.48, 0.70)
    ax.legend(fontsize=9)
    ax.set_title("More cross-org bridges -> stronger signal")
    fig.savefig(FIG / "fig_pcross.pdf")
    plt.close(fig)
    print("[fig] fig_pcross.pdf")


def fig_fabrication():
    df = pd.read_csv(RESULTS / "exp_sweep_fab.csv")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(
        df["fabrication"], df["full_auc"], "o-", c="#1f77b4", label="full multiplex"
    )
    ax.plot(
        df["fabrication"],
        df["shuffled_auc"],
        "s--",
        c="#d62728",
        label="control (shuffled)",
    )
    ax.fill_between(
        df["fabrication"],
        df["shuffled_auc"],
        df["full_auc"],
        alpha=0.12,
        color="#1f77b4",
    )
    ax.set_xlabel("rate of OSINT-trace fabrication by the attacker")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.48, 0.90)
    ax.legend(fontsize=9)
    ax.set_title("Adversarial crossover: fake traces erode the advantage")
    fig.savefig(FIG / "fig_fabrication.pdf")
    plt.close(fig)
    print("[fig] fig_fabrication.pdf")


def fig_overlap():
    df = pd.read_csv(RESULTS / "layers" / "layer_overlap.csv")
    layers = sorted(set(df["layer_a"]) | set(df["layer_b"]))
    idx = {l: i for i, l in enumerate(layers)}
    M = np.eye(len(layers))
    for _, r in df.iterrows():
        i, j = idx[r["layer_a"]], idx[r["layer_b"]]
        M[i, j] = M[j, i] = r["jaccard_edges"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1)
    short = [l.replace("osint_", "").replace("_", " ")[:12] for l in layers]
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(short, fontsize=8)
    for i in range(len(layers)):
        for j in range(len(layers)):
            ax.text(
                j,
                i,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if M[i, j] < 0.6 else "white",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, label="Jaccard (edge sets)")
    ax.set_title("Layer correlation (contacts ~ hierarchy ~ domain)")
    fig.savefig(FIG / "fig_overlap.pdf")
    plt.close(fig)
    print("[fig] fig_overlap.pdf")


def fig_layer_selection():
    df = pd.read_csv(RESULTS / "exp_layer_selection.csv")
    fwd = df[df["section"] == "forward_cumulative_auc"].copy()
    fwd["ord"] = fwd["layer"].str.split(":").str[0].astype(int)
    fwd["name"] = fwd["layer"].str.split(":").str[1]
    fwd = fwd.sort_values("ord")
    loo = df[df["section"] == "leave_one_out_drop"].copy()
    loo["value"] = loo["value"].astype(float)
    loo = loo.sort_values("value", ascending=False).head(8)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7.4))
    ax1.plot(range(1, len(fwd) + 1), fwd["value"].astype(float), "o-", c="#1f77b4")
    ax1.set_xticks(range(1, len(fwd) + 1))
    ax1.set_xticklabels(fwd["name"], rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("AUC (cumulative)")
    ax1.set_title("Forward selection: how many layers suffice")
    ax1.axhline(
        float(df[df.section == "full"]["value"].iloc[0]), ls=":", c="grey", lw=0.8
    )
    cols = ["#d62728" if v >= 0.01 else "#999999" for v in loo["value"]]
    ax2.barh(range(len(loo)), loo["value"], color=cols)
    ax2.set_yticks(range(len(loo)))
    ax2.set_yticklabels(loo["layer"], fontsize=7)
    ax2.invert_yaxis()
    ax2.axvline(0.01, ls=":", c="grey", lw=0.8)
    ax2.set_xlabel("AUC drop after removing a layer")
    ax2.set_title("Leave-one-out: which layers are necessary")
    fig.tight_layout()
    fig.savefig(FIG / "fig_layer_selection.pdf")
    plt.close(fig)
    print("[fig] fig_layer_selection.pdf")


def fig_layer_by_threat():
    df = pd.read_csv(RESULTS / "exp_layer_by_threat.csv")
    tasks = [
        ("nowi_nadawcy", "New senders (FP)"),
        ("przejecia", "Compromised accounts (FN)"),
    ]
    cfgs = [
        ("tylko_OSINT", "OSINT only", "#2ca02c"),
        ("tylko_czas", "time only", "#ff7f0e"),
        ("full", "full", "#1f77b4"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, (tk, tlab) in zip(axes, tasks):
        sub = df[df["task"] == tk].set_index("config")["auc"]
        vals = [sub.get(c, 0) for c, _l, _col in cfgs]
        ax.bar(range(len(cfgs)), vals, color=[c for _k, _l, c in cfgs])
        ax.axhline(0.5, ls=":", c="grey", lw=0.8)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels([l for _k, l, _c in cfgs], fontsize=8)
        ax.set_title(tlab, fontsize=10)
        ax.set_ylim(0.45, 0.95)
    axes[0].set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig(FIG / "fig_layer_by_threat.pdf")
    plt.close(fig)
    print("[fig] fig_layer_by_threat.pdf")


def fig_sensitivity():
    specs = [
        ("offhours", "off_hours", "fraction of legitimate mail off-rhythm"),
        ("mimic", "mimic", "fraction of attacks mimicking the rhythm"),
        ("N", "N", "population size N"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (suf, xcol, xlab) in zip(axes, specs):
        df = pd.read_csv(RESULTS / f"exp_sensitivity_{suf}.csv")
        ax.plot(df[xcol], df["full_auc"], "o-", c="#1f77b4", label="full")
        ax.plot(df[xcol], df["static_auc"], "s--", c="#2ca02c", label="static")
        ax.plot(df[xcol], df["contact_auc"], "^:", c="#888888", label="contacts")
        ax.axhline(0.5, ls=":", c="grey", lw=0.6)
        ax.set_xlabel(xlab)
        ax.set_ylim(0.45, 0.92)
        ax.set_title(xlab, fontsize=9)
    axes[0].set_ylabel("AUC")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_sensitivity.pdf")
    plt.close(fig)
    print("[fig] fig_sensitivity.pdf")


def fig_enron_curve():
    df = pd.read_csv(RESULTS / "exp_enron_multiplex.csv")
    fprs = [float(c.split("fpr")[1]) for c in df.columns if c.startswith("recall@fpr")]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    styles = {
        "full": ("o-", "#1f77b4", "full multiplex"),
        "contact": ("s--", "#888888", "contacts only"),
    }
    for _, row in df.iterrows():
        st, c, lab = styles.get(row["model"], ("o-", "k", row["model"]))
        rec = [row[f"recall@fpr{fp}"] for fp in fprs]
        ax.plot(fprs, rec, st, c=c, label=f"{lab} (AUC {row['auc']:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("FPR (log scale)")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Sensitivity-FPR curve on the real Enron graph")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(FIG / "fig_enron_curve.pdf")
    plt.close(fig)
    print("[fig] fig_enron_curve.pdf")


def fig_enron_audit():
    df = pd.read_csv(RESULTS / "exp_enron_leak_audit.csv")
    rates = sorted(df["off_hours_rate"].unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.7))
    for m, c, lab in [
        ("full", "#1f77b4", "full multiplex"),
        ("tc_only", "#ff7f0e", "time only"),
    ]:
        d = df[df.model == m].set_index("off_hours_rate")
        a1.plot(
            [r * 100 for r in rates],
            [d.loc[r, "auc"] for r in rates],
            "o-",
            c=c,
            label=lab,
        )
        a2.plot(
            [r * 100 for r in rates],
            [d.loc[r, "recall_fpr1"] for r in rates],
            "o-",
            c=c,
            label=lab,
        )
    a1.set_ylabel("AUC")
    a1.set_title("AUC (gentle degradation)")
    a1.set_ylim(0.6, 0.96)
    a2.set_ylabel("Sensitivity@FPR=1%")
    a2.set_title("Sensitivity@FPR=1% (sharp drop = leakage)")
    for ax in (a1, a2):
        ax.set_xlabel("legitimate mail off-rhythm [%]")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_enron_audit.pdf")
    plt.close(fig)
    print("[fig] fig_enron_audit.pdf")


def main():
    fig_ablation()
    fig_pcross()
    fig_fabrication()
    fig_overlap()
    fig_layer_selection()
    fig_layer_by_threat()
    fig_sensitivity()
    try:
        fig_enron_curve()
    except FileNotFoundError:
        print("[fig] (skipping enron_curve — missing csv)")
    try:
        fig_enron_audit()
    except FileNotFoundError:
        print("[fig] (skipping enron_audit — missing csv)")
    print(f"[fig] all figures -> {FIG}")


if __name__ == "__main__":
    main()
