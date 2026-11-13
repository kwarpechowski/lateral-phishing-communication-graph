# -*- coding: utf-8 -*-
"""R4-A4: generator tabel LaTeX bezposrednio z CSV (rozbudowa A6 z rundy 3).

Cel: nikt nie przepisuje liczb recznie. Skrypt czyta results/*.csv (srednie)
i results/ci/*_ci.csv (bootstrap CI z analysis/boot_ci.py) i emituje ciala
tabular do code/gen_tables/*.tex.

Format CI (R4-A4):
  * tabele w glownym tekście (Tab. 3, 4, 5, 6 A): srednia +/- polowa szerokosci
    przedzialu 95% na kolumnie AUC,
  * tabele w suplemencie (S2 sweep, S4 label noise): pelny przedzial [lo; hi].

Uruchomienie: python analysis/gen_tables.py  (po fali eksperymentow + boot_ci.py)
Wyjscie: code/gen_tables/*.tex — artykuly robia \csname @@input\endcsname{...}
"""

from __future__ import annotations

import csv
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
RESULTS = CODE / "results"
OUT = CODE / "gen_tables"
OUT.mkdir(exist_ok=True)

LABELS = {
    "tab_1hop": "1-hop (tabular)",
    "gnn_static": "Static GNN",
    "tab_ctx": "Hand-crafted context (tab.)",
    "compa": "Volumetric (COMPA)",
    "gnn_temporal": "Temporal GNN",
    "gnn_temporal_shuf": "Temporal (time shuffled)",
    "reczny-kontekst": "manual",
    "hopper": "path",
    "hopper-bez-precursor": "path (no precursor)",
    "temporalny-GNN": "temp.",
    "COMPA": "COMPA",
    "rhythm_only": "rhythm-only",
    "tgn_pyg": "Library TGN (PyG, 3 seeds)",
    "gnn_gru": "TGN-lite (GRU)",
    "gnn_attn": "TGAT-lite (attn)",
}


def _load(name):
    with (RESULTS / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f.open if False else f))


def _fmt(v, bold=False):
    try:
        x = f"${float(v):.2f}$"
    except (TypeError, ValueError):
        x = str(v)
    return f"\\textbf{{{x}}}" if bold else x


def _ci(name, keycols, metric="auc"):
    """Wczytaj results/ci/<name>_ci.csv -> dict {tuple(keycols): (mean, lo, hi)}."""
    p = RESULTS / "ci" / (name + "_ci.csv")
    if not p.exists():
        return {}
    out = {}
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("metric") != metric:
                continue
            try:
                out[tuple(str(r[c]) for c in keycols)] = (
                    float(r["mean"]),
                    float(r["ci_lo"]),
                    float(r["ci_hi"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _fmt_pm(ci3, bold=False):
    """Glowny tekst: srednia +/- polowa szerokosci CI95."""
    if ci3 is None:
        return "--"
    mean, lo, hi = ci3
    hw = (hi - lo) / 2.0
    x = f"${mean:.2f}{{\\pm}}{hw:.2f}$"
    return f"\\textbf{{{x}}}" if bold else x


def _fmt_int(ci3, bold=False):
    """Suplement: pelny przedzial [lo; hi] obok sredniej."""
    if ci3 is None:
        return "--"
    mean, lo, hi = ci3
    x = f"${mean:.2f}$\\,[{lo:.2f};{hi:.2f}]"
    return f"\\textbf{{{x}}}" if bold else x


def detectors():
    """Tab. 3 (A): hierarchia detektorow na grafie case study (+ CI95 z per-seed)."""
    rows = _load("exp_cascade.csv")
    tgn = _load("exp_cascade_tgn.csv")
    ci = _ci("exp_cascade", ["model"])
    by = {r["model"]: r for r in rows}
    by["tgn_pyg"] = next((r for r in tgn if r.get("model") == "tgn_pyg"), None)
    order = [
        "tab_1hop",
        "gnn_static",
        "tab_ctx",
        "compa",
        "tgn_pyg",
        "gnn_temporal",
        "gnn_temporal_shuf",
    ]
    lines = []
    best_auc = max(
        float(by[m]["auc"]) for m in order if m in by and by[m] and (m,) in ci
    )
    for m in order:
        if m not in by or by[m] is None:
            continue
        r = by[m]
        b = (m,) in ci and float(r["auc"]) == best_auc
        pm = _fmt_pm(ci.get((m,)), bold=b) if (m,) in ci else _fmt(r["auc"], b)
        lines.append(f"{LABELS[m]} & {pm} & {_fmt(r['recall_fpr1'], False)} \\\\")
    (OUT / "tab_detectors.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_detectors.tex")


def stealth():
    """Tab. 4 (A): atak skryty, 20 seedow, K-sweep (timing=det) + CI95 na AUC."""
    rows = _load("exp_cascade_stealth.csv")
    ci = _ci("exp_cascade_stealth", ["timing", "K", "model"])
    ks = [
        ("1", "$K{=}1$ (extremely stealthy)"),
        ("2", "$K{=}2$"),
        ("3", "$K{=}3$"),
        ("5", "$K{=}5$"),
        ("999", "blast"),
    ]
    by = {(r["K"], r["model"]): r for r in rows if r.get("timing", "det") == "det"}
    lines = []
    for k, lab in ks:
        vol = by.get((k, "compa"), {})
        man = by.get((k, "tab_ctx"), {})
        tmp = by.get((k, "gnn_temporal"), {})
        aucs = {"v": float(vol["auc"]), "m": float(man["auc"]), "t": float(tmp["auc"])}
        r1s = {
            "v": float(vol["recall_fpr1"]),
            "m": float(man["recall_fpr1"]),
            "t": float(tmp["recall_fpr1"]),
        }
        ba = max(aucs.values())
        br = max(r1s.values())

        def cell(key):
            return _fmt_pm(
                ci.get(
                    ("det", k, {"v": "compa", "m": "tab_ctx", "t": "gnn_temporal"}[key])
                ),
                bold=aucs[key] == ba,
            )

        lines.append(
            f"{lab} & {cell('v')} & {cell('m')} & {cell('t')} & "
            f"{_fmt(r1s['v'])} & {_fmt(r1s['m'], r1s['m'] == br)} & "
            f"{_fmt(r1s['t'], r1s['t'] == br)} \\\\"
        )
    (OUT / "tab_stealth.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_stealth.tex")


def baselines():
    """Tab. 5 (A): baselines (a) stealth sweep (b) surgical (timing=det) + CI AUC."""
    st = [r for r in _load("cb_hopper_stealth.csv") if r.get("timing", "det") == "det"]
    su = _load("cb_surgical_fanout.csv")
    ci = _ci("cb_hopper_stealth", ["timing", "poziom", "detektor"])
    ci_s = _ci("cb_surgical_fanout", ["fanout", "detektor"])
    lines = []
    names = [
        "auc_COMPA",
        "auc_hopper",
        "auc_hopper-bez-precursor",
        "auc_reczny-kontekst",
        "auc_temporalny-GNN",
    ]
    dets = [
        "COMPA",
        "hopper",
        "hopper-bez-precursor",
        "reczny-kontekst",
        "temporalny-GNN",
    ]
    for r in st:
        vals = [float(r[n]) for n in names]
        mx = max(vals)
        cells = []
        for d, n in zip(dets, names):
            c = ci.get(("det", r["poziom"], d))
            cells.append(
                _fmt_pm(c, bold=vals[names.index(n)] == mx)
                if c
                else _fmt(r[n], vals[names.index(n)] == mx)
            )
        reach = r.get("zasieg", "")
        reach_pct = round(float(reach) * 100) if reach else 0
        lab = r["poziom"].replace("K=", "K=")
        lines.append(f"{lab} ({reach_pct}\\%) & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append(
        "\\multicolumn{6}{@{}l}{\\emph{(b) Surgical attack without "
        "propagation: $p_\\mathrm{inf}{=}0$, by fan-out}} \\\\"
    )
    for r in su:
        vals = [float(r[n]) for n in names]
        mx = max(vals)
        cells = []
        for d, n in zip(dets, names):
            c = ci_s.get((r["fanout"], d))
            cells.append(
                _fmt_pm(c, bold=vals[names.index(n)] == mx)
                if c
                else _fmt(r[n], vals[names.index(n)] == mx)
            )
        lines.append(f"fan-out {r['fanout']} & " + " & ".join(cells) + " \\\\")
    (OUT / "tab_baselines.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_baselines.tex")


def realgraphs():
    """Tab. 6 (A): Enron + SNAP; stealthy = timing EMP (odstepy z tla — R4-A2:
    glowna tabela rezymu stealth po usunieciu sygnatury iniekcji), blast = det."""
    en = _load("exp_cascade_enron.csv")
    sn = _load("exp_cascade_realgraphs.csv")
    ci_e = _ci("exp_cascade_enron", ["rezym", "timing", "model"])
    ci_s = _ci("exp_cascade_realgraphs", ["dataset", "regime", "timing", "model"])
    lines = []
    by = {
        (r["rezym"], r["model"]): r
        for r in en
        if r.get("timing", "det") == ("det" if r["rezym"] == "blast" else "emp")
    }
    for regime, rlab in [("blast", "naive blast"), ("stealthy", "stealthy attacker")]:
        tim_key = "det" if regime == "blast" else "emp"
        lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{\\quad {rlab}}}}} \\\\")
        det = [
            ("tab_1hop", "1-hop (tabular)"),
            ("tab_ctx", "Hand-crafted context (tab.)"),
            ("compa", "Volumetric (COMPA)"),
            ("gnn_temporal", "Temporal GNN"),
            ("gnn_temporal_shuf", "Temporal (time shuffled)"),
        ]
        best = max(
            float(by[(regime, d[0])]["auc"]) for d in det if (regime, d[0]) in by
        )
        for key, lab in det:
            if (regime, key) not in by:
                continue
            r = by[(regime, key)]
            b = float(r["auc"]) == best
            c = ci_e.get((regime, tim_key, key))
            auc_cell = _fmt_pm(c, bold=b) if c else _fmt(r["auc"], b)
            lines.append(
                f" & {lab} & {auc_cell} & {_fmt(r['recall_fpr1'], False)} \\\\"
            )
    lines.append("\\midrule")
    lines.append(
        "\\multicolumn{4}{@{}l}{\\emph{(b) SNAP graphs, naive blast vs "
        "stealthy attacker ($\\mathbf{emp}$ timing), $10$~seeds}} \\\\"
    )
    by2 = {
        (r["dataset"], r["regime"], r["model"]): r
        for r in sn
        if r.get("timing", "det") == ("det" if r["regime"] == "blast" else "emp")
    }
    for ds in ["email-Eu-core", "CollegeMsg"]:
        dsl = f"\\texttt{{{ds}}}" if ds != "CollegeMsg" else "\\texttt{CollegeMsg}"
        for regime, rlab in [("blast", "blast"), ("stealthy", "stealthy")]:
            tim_key = "det" if regime == "blast" else "emp"
            det = [
                ("compa", "volumetric"),
                ("tab_ctx", "hand-crafted context"),
                ("gnn_temporal", "temporal GNN"),
            ]
            best = max(
                float(by2[(ds, regime, key)]["auc"])
                for key, _l in det
                if (ds, regime, key) in by2
            )
            for j, (key, lab) in enumerate(det):
                if (ds, regime, key) not in by2:
                    continue
                r = by2[(ds, regime, key)]
                b = float(r["auc"]) == best
                c = ci_s.get((ds, regime, tim_key, key))
                auc_cell = _fmt_pm(c, bold=b) if c else _fmt(r["auc"], b)
                pre = f"\\multirow{{3}}{{*}}{{{dsl} ({rlab})}}" if j == 0 else ""
                lines.append(
                    f"{pre} & {lab} & {auc_cell} & {_fmt(r['recall_fpr1'], False)} \\\\"
                )
    (OUT / "tab_realgraphs.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_realgraphs.tex (stealthy=emp)")


def sweep():
    """Tab. S2 (A): sweep p_infect (suplement: pelne CI [lo;hi])."""
    rows = _load("exp_cascade_sweep.csv")
    ci = _ci("exp_cascade_sweep", ["sweep", "param", "model"])
    lines = []
    order = ["tab_1hop", "tab_ctx", "compa", "gnn_temporal"]
    for v in ["0.05", "0.1", "0.2", "0.3", "0.5", "0.7", "0.9"]:
        by = {
            r["model"]: r
            for r in rows
            if r["sweep"] == "p_infect" and float(r["value"]) == float(v)
        }
        vals = [float(by[m]["auc"]) for m in order if m in by]
        mx = max(vals) if vals else 0.0
        cells = []
        for m in order:
            c = ci.get(("p_infect", str(float(v)), m))
            cells.append(_fmt_int(c, bold=(m in by and float(by[m]["auc"]) == mx)))
        lines.append(f"${v}$ & " + " & ".join(cells) + " \\\\")
    (OUT / "tab_sweep.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_sweep.tex")


def labelnoise():
    """Tab. S4 (A): szum etykiet (etykiety 5/10/20%, suplement: pelne CI)."""
    rows = _load("cb_labelnoise.csv")
    ci = _ci("cb_labelnoise", ["szum", "detektor"])
    names = [
        "COMPA",
        "hopper",
        "hopper-bez-precursor",
        "reczny-kontekst",
        "temporalny-GNN",
    ]
    lab_by = {"0.0": "$0\\%$", "0.05": "$5\\%$", "0.1": "$10\\%$", "0.2": "$20\\%$"}
    lines = []
    for r in rows:
        vals = [float(r[n]) for n in names]
        mx = max(vals)
        cells = []
        for n in names:
            c = ci.get((r["szum"], n))
            cells.append(_fmt_int(c, bold=float(r[n]) == mx))
        lines.append(
            f"{lab_by.get(r['szum'], r['szum'])} & " + " & ".join(cells) + " \\\\"
        )
    (OUT / "tab_labelnoise.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_labelnoise.tex")


def enronsplit():
    """Tab. IV (B): kontrola wycieku rytmu (z CI z per-seed splitu)."""
    rows = _load("exp_enron_temporal_split.csv")
    ci = _ci("exp_enron_temporal_split", ["mode", "metric"])
    lines = []
    for mode, lab in [
        ("leaky", "full history (\\emph{with leakage})"),
        ("split", "early window (\\emph{temporal split})"),
        ("split_oh", "early window $+\\,$off-hours (\\emph{both})"),
    ]:
        by = {r["metric"]: r for r in rows if r["mode"] == mode}
        if not by:
            continue
        ta = by.get("temporal_auc", {})
        tr = by.get("temporal_r1", {})
        fa = by.get("full_auc", {})
        fr = by.get("full_r1", {})
        ca = _fmt_pm(ci.get((mode, "temporal_auc")))
        cr = _fmt(by.get("temporal_r1", {}).get("mean", "--"))
        fa_c = _fmt_pm(ci.get((mode, "full_auc")))
        fr_c = _fmt(by.get("full_r1", {}).get("mean", "--"))
        lines.append(f"{lab} & {ca} & {cr} & {fa_c} & {fr_c} \\\\")
    (OUT / "tab_enronsplit.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_enronsplit.tex")


def fabrication():
    """R4-B5: krzywa fabrykacji (B) — AUC/shuffle/FP nowych nadawcow."""
    rows = _load("exp_multiplex.csv")
    by = {r["config"]: r for r in rows}
    lines = []
    for cfg in ["default", "fab10", "fab25", "fab50", "fab75", "fab100"]:
        if cfg not in by:
            continue
        r = by[cfg]
        lines.append(
            f"{cfg.replace('fab', '').replace('default', '0')}\\% & "
            f"{_fmt(r['mux_auc'])} & {_fmt(r['shuf_auc'])} & "
            f"{_fmt(r['contact_auc'])} & "
            f"{_fmt(float(r['fp_newsender_contact']) * 100)} & "
            f"{_fmt(float(r['fp_newsender_mux']) * 100)} \\\\"
        )
    (OUT / "tab_fabrication.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_fabrication.tex")


# ============ R4-B5: tabele B (Tab. II-VI) z CSV ============


def inv1():
    """Tab. II (B): cross-layer inconsistency (exp_multiplex 'default')."""
    by = {r["config"]: r for r in _load("exp_multiplex.csv")}
    r = by.get("default")
    if not r:
        return
    mx = max(float(r["contact_auc"]), float(r["mux_auc"]))
    lines = [
        f"Contacts only & {_fmt(r['contact_auc'], float(r['contact_auc']) == mx)} & "
        f"{_fmt(float(r['fp_newsender_contact']) * 100, False)} \\\\",
        f"Multiplex $+$ rhythm & {_fmt(r['mux_auc'], float(r['mux_auc']) == mx)} & "
        f"{_fmt(float(r['fp_newsender_mux']) * 100, True)} \\\\",
        f"Multiplex (shuffled) & {_fmt(r['shuf_auc'], False)} & --- \\\\",
    ]
    (OUT / "tab_inv1.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_inv1.tex")


def topo():
    """Tab. III (B): topology decorrelation."""
    rows = _load("exp_topology_correlation.csv")
    lines = []
    for r in rows:
        rho = float(r["rho"])
        lab = f"${rho:.1f}$" + (
            " (dense)" if rho == 0 else " (decorrel.)" if rho >= 0.8 else ""
        )
        lines.append(
            f"{lab} & {_fmt(r['jaccard_struct'])} & {_fmt(r['loo_struct'])} & "
            f"{_fmt(r['contact_auc'])} \\\\"
        )
    (OUT / "tab_topo.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_topo.tex")


def susc():
    """Tab. V (B): susceptibility z bootstrap CI z per-seed."""
    rows = _load("exp_susceptibility.csv")
    ci = _ci("exp_susceptibility", ["group"], metric="recall_fpr1")
    order = [
        ("channel:bec_off", " & External impers.\\ (cold BEC)"),
        ("channel:compromised", " & Compromised acct (lateral)"),
        ("exec", " & Executive"),
        ("senior", " & Senior"),
        ("mid", " & Mid-level"),
        ("junior", " & Junior"),
        ("bec_exec", " & Executive"),
        ("bec_senior", " & Senior"),
        ("bec_mid", " & Mid-level"),
        ("bec_junior", " & Junior"),
    ]
    by = {r["group"]: r for r in rows}
    lines = []
    for k, lab in order:
        if k not in by:
            continue
        r = by[k]
        c = ci.get((k,))
        cell = _fmt_pm(c) if c else _fmt(r["recall_fpr1"])
        lines.append(f"{lab} & {cell} & {_fmt(r['susc'])} \\\\")
    (OUT / "tab_susc.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_susc.tex")


def _fmt3(v, bold=False):
    """3 miejsca po przecinku (content fusion: 0.997 vs 0.998)."""
    try:
        x = f"${float(v):.3f}$"
    except (TypeError, ValueError):
        x = str(v)
    return f"\\textbf{{{x}}}" if bold else x


def content():
    """Tab. VI (B): fusion graf + tresc."""
    rows = _load("exp_content_fusion.csv")
    by = {r["regime"]: r for r in rows}
    lines = []
    for reg, lab in [
        ("tell", "Tell-tale (content trace)"),
        ("mimicry", "Mimicry (BEC)"),
    ]:
        if reg not in by:
            continue
        r = by[reg]
        mx = max(float(r["content_auc"]), float(r["mux_auc"]), float(r["fusion_auc"]))
        lines.append(
            f"{lab} & {_fmt3(r['content_auc'], float(r['content_auc']) == mx)} & "
            f"{_fmt3(r['mux_auc'], float(r['mux_auc']) == mx)} & "
            f"{_fmt3(r['fusion_auc'], float(r['fusion_auc']) == mx)} \\\\"
        )
    (OUT / "tab_content.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_content.tex")


if __name__ == "__main__":
    detectors()
    stealth()
    baselines()
    realgraphs()
    sweep()
    labelnoise()
    enronsplit()
    fabrication()
    inv1()
    topo()
    susc()
    content()
