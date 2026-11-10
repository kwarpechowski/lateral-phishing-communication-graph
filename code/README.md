# Experiment code

Deterministic, seeded experiments. Each script under `experiments/` writes a CSV to `results/`
and corresponds to a table/figure in the paper.

## Modules

| Module | Role |
|--------|------|
| `graph/` | Builds the **domain knowledge graph**: contact/structural layers, eight OSINT layers from persona fields `c3`–`c8`, and the temporal-rhythm overlay (Invariant I). |
| `data/` | Synthetic-population generators: `org_graph.py` (org graph, roles, seniority) and `twin_generator.py` (persona schema `c1`–`c8`). |
| `cascadebench/` | Leak-aware cascade testbed: cascade model, detector panel (1-hop, hand-crafted context, static GNN, temporal GNN/TGN-lite, volumetric, path-based), topology pool, statistics (bootstrap CI, Wilcoxon, Cliff's δ). Drives Invariant II. |
| `experiments/` | Runnable experiments (below). |
| `analysis/` | Plotting scripts that turn `results/*.csv` into the figures. |

## Experiments → paper

| Script (`experiments/`) | Produces |
|-------------------------|----------|
| `exp_multiplex.py` | Invariant I: multiplex vs contacts (Tab. *tab:inv1*) |
| `exp_consistency.py` | Explicit consistency operator vs concatenation (Tab. *tab:consistency*) |
| `exp_layer_selection.py` | Forward / leave-one-out layer selection (Fig. *fig:layersel*) |
| `exp_layer_by_threat.py` | Per-threat complementarity (Fig. *fig:bythreat*) |
| `exp_extended.py` | Ablation + sensitivity sweeps (Figs. *fig:ablation*, *fig:sensitivity*) |
| `exp_fusion.py` | Unified event model, Recall@FPR=1% by attack channel |
| `exp_susceptibility.py` | **Organizational susceptibility** (Tab. *tab:susc*) |
| `exp_cascade.py` | Invariant II: detector hierarchy (Tab. *tab:detectors*, Fig. *fig:cascade_detectors*) |
| `exp_cascade_stealth.py` | Stealthy attacker; volumetric collapse (Tabs. *tab:stealth*, *tab:hopper*, *tab:blindspot*) |
| `exp_cascade_sweep.py` | Cascade-model sweep p_inf / depth (Tab. *tab:pinf*) |
| `exp_cascade_arch.py`, `exp_cascade_tgn.py` | Architecture robustness (TGN-lite / TGAT-lite / library TGN) |
| `exp_enron_multiplex.py`, `exp_enron_temporal_split.py`, `exp_enron_leak_audit.py` | Enron real-graph validation + leak controls (Tab. *tab:enronsplit*, Fig. *fig:enron_audit*) |
| `exp_cascade_enron.py` | Enron cascade hierarchy (Tab. *tab:enron-casc*) |
| `exp_cascade_realgraphs.py` | Transfer to SNAP graphs (Tab. *tab:realgraphs*) |
| `exp_cascade_fpr.py` | Sensitivity–FPR curve + significance (Fig. *fig:cascade_fpr*, Tab. *tab:headline-ci*) |
| `exp_content_fusion.py` | Graph + content fusion (Tab. *tab:content*) |
| `exp_p3_*.py` | Leak-aware methodology studies (community-label sweep, multi-strategy transfer, grid search, predictivity). |

Run, e.g.:
```bash
# CPU-friendly (LightGBM); fast
python experiments/exp_susceptibility.py
# GNN-heavy (torch); cap threads to avoid oversubscription
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python experiments/exp_cascade.py
```
The synthetic population is regenerated deterministically; no download is needed for the synthetic results.

## Bundled generated/computed artifacts

- `data/twins/` — 150 synthetic personas (the generated population), schema `c1`–`c8`.
- `results/` — every experiment output (CSVs, `results/layers/`, `results/osint_overlay/`,
  `twin_self.csv`, corpora). The synthetic results and all paper figures reproduce from these
  with no download. Regenerate the population with `python data/twin_generator.py` (deterministic, seeded).

## Input data (not bundled — public, ~0.9 GB)

Download and place under `data/` (`data/realgraphs/`, `data/elliptic/`; `data/topo_tmp/` is a regenerable cache):

- **Enron email corpus** (maildir headers) — https://www.cs.cmu.edu/~enron/
- **SNAP `email-Eu-core` (temporal)** — https://snap.stanford.edu/data/email-Eu-core-temporal.html
- **SNAP `CollegeMsg`** — https://snap.stanford.edu/data/CollegeMsg.html
- **Elliptic** (optional, used only by `exp_p3_elliptic*.py`) — public Bitcoin transaction-graph dataset.

We use only the **real topology / timestamps**; attack labels are controlled injections
(matched-design), so the corpora carry no labeled phishing.
