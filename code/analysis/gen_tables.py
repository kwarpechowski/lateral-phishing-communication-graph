# -*- coding: utf-8 -*-
"""A6: generator tabel LaTeX bezposrednio z CSV (recenzja wewnetrzna r2).

Cel: nikt nie przepisuje liczb recznie. Skrypt czyta results/*.csv i emituje
calkowite srodowiska tabular do code/gen_tables/*.tex (bez float/caption —
podpisy zostaja w tekscie artykulu, liczby przychodza z danych).

Uruchomienie: python analysis/gen_tables.py  (po zakonczeniu fali eksperymentow)
Wyjscie: code/gen_tables/*.tex — artykuly robiia \input{...}
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
RESULTS = CODE / "results"
OUT = CODE / "gen_tables"
OUT.mkdir(exist_ok=True)

LABELS = {
    "tab_1hop": "1-hop (tabular)",
    "gnn_static": "Static GNN",
    "tab_ctx": "Hand-crafted context (tab.)",
    "gnn_temporal": "Temporal GNN",
    "gnn_temporal_shuf": "Temporal (time shuffled)",
    "compa": "vol.",
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
        return list(csv.DictReader(f))


def _fmt(v, bold=False):
    try:
        x = f"{float(v):.3f}".lstrip("0") if abs(float(v)) < 1 else f"{float(v):.2f}"
        x = f"{float(v):.2f}"
        x = f"${x}$"
    except (TypeError, ValueError):
        x = str(v)
    return f"\\textbf{{{x}}}" if bold else x


def _bold_best(rows, cols, get):
    """Indeksy najlepszych wartosci w kazdym wierszu (do pogrubienia)."""
    best = []
    for r in rows:
        vals = [(c, get(r, c)) for c in cols if get(r, c) is not None]
        if not vals:
            best.append(None)
            continue
        mx = max(v for _c, v in vals)
        best.append([c for c, v in vals if v == mx])
    return best


def detectors():
    """Tab. 3 (A): hierarchia detektorow na grafie case study."""
    rows = _load("exp_cascade.csv")
    tgn = _load("exp_cascade_tgn.csv")
    by = {r["model"]: r for r in rows}
    by["tgn_pyg"] = next((r for r in tgn if r.get("model") == "tgn_pyg"), None)
    order = [
        "tab_1hop",
        "gnn_static",
        "tab_ctx",
        "tgn_pyg",
        "gnn_temporal",
        "gnn_temporal_shuf",
    ]
    lines = []
    best_auc = max(float(by[m]["auc"]) for m in order if m in by and by[m])
    for m in order:
        if m not in by or by[m] is None:
            continue
        r = by[m]
        lines.append(
            f"{LABELS[m]} & {_fmt(r['auc'], float(r['auc']) == best_auc)} & "
            f"{_fmt(r['recall_fpr1'], False)} \\\\"
        )
    (OUT / "tab_detectors.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_detectors.tex")


def stealth():
    """Tab. 4 (A): atak skryty, 20 seedow, K-sweep."""
    rows = _load("exp_cascade_stealth.csv")
    ks = [
        ("1", "$K{=}1$ (extremely stealthy)"),
        ("2", "$K{=}2$"),
        ("3", "$K{=}3$"),
        ("5", "$K{=}5$"),
        ("999", "blast"),
    ]
    by = {(r["K"], r["model"]): r for r in rows}
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
        lines.append(
            f"{lab} & {_fmt(aucs['v'])} & {_fmt(aucs['m'], aucs['m'] == ba)} & "
            f"{_fmt(aucs['t'], aucs['t'] == ba)} & {_fmt(r1s['v'])} & "
            f"{_fmt(r1s['m'], r1s['m'] == br)} & {_fmt(r1s['t'], r1s['t'] == br)} \\\\"
        )
    (OUT / "tab_stealth.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_stealth.tex")


def baselines():
    """Tab. 5 (A): baselines (a) stealth sweep (b) surgical."""
    st = _load("cb_hopper_stealth.csv")
    su = _load("cb_surgical_fanout.csv")
    lines = []
    names = [
        "auc_COMPA",
        "auc_hopper",
        "auc_hopper-bez-precursor",
        "auc_reczny-kontekst",
        "auc_temporalny-GNN",
    ]
    for r in st:
        k = r["poziom"] if "poziom" in r else r.get("K", r.get("poziom", "?"))
        vals = [float(r[n]) for n in names]
        mx = max(vals)
        cells = " & ".join(_fmt(v, v == mx) for v in vals)
        reach = r.get("zasieg", "")
        reach_pct = round(float(reach) * 100) if reach else 0
        lines.append(f"{k} ({reach_pct}\\%) & {cells} \\\\")
    lines.append("\\midrule")
    lines.append(
        "\\multicolumn{6}{@{}l}{\\emph{(b) Surgical attack without "
        "propagation: $p_\\mathrm{inf}{=}0$, by fan-out}} \\\\"
    )
    for r in su:
        vals = [float(r[n]) for n in names]
        mx = max(vals)
        cells = " & ".join(_fmt(v, v == mx) for v in vals)
        lines.append(f"fan-out {r['fanout']} & {cells} \\\\")
    (OUT / "tab_baselines.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_baselines.tex")


def realgraphs():
    """Tab. 6 (A): Enron (oba rezymy) + SNAP."""
    en = _load("exp_cascade_enron.csv")
    sn = _load("exp_cascade_realgraphs.csv")
    lines = []
    by = {(r["rezym"], r["model"]): r for r in en}
    for regime, rlab in [("blast", "naive blast"), ("stealthy", "stealthy attacker")]:
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
            a = float(r["auc"])
            lines.append(
                f" & {lab} & {_fmt(a, a == best)} & "
                f"{_fmt(r['recall_fpr1'], False)} \\\\"
            )
    lines.append("\\midrule")
    lines.append(
        "\\multicolumn{4}{@{}l}{\\emph{(b) SNAP graphs, naive blast vs "
        "stealthy attacker, $3$~seeds}} \\\\"
    )
    by2 = {(r["dataset"], r["regime"], r["model"]): r for r in sn}
    for ds in ["email-Eu-core", "CollegeMsg"]:
        dsl = f"\\texttt{{{ds}}}" if ds != "CollegeMsg" else "\\texttt{CollegeMsg}"
        for regime, rlab in [("blast", "blast"), ("stealthy", "stealthy")]:
            det = [
                ("compa", "volumetric"),
                ("tab_ctx", "hand-crafted context"),
                ("gnn_temporal", "temporal GNN"),
            ]
            best = max(
                float(by2[(ds, regime, d[0])]["auc"])
                for d in det
                if (ds, regime, d[0]) in by2
            )
            for j, (key, lab) in enumerate(det):
                if (ds, regime, key) not in by2:
                    continue
                r = by2[(ds, regime, key)]
                a = float(r["auc"])
                pre = f"\\multirow{{3}}{{*}}{{{dsl} ({rlab})}}" if j == 0 else ""
                lines.append(
                    f"{pre} & {lab} & {_fmt(a, a == best)} & "
                    f"{_fmt(r['recall_fpr1'], False)} \\\\"
                )
    (OUT / "tab_realgraphs.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_realgraphs.tex")


def enronsplit():
    """Tab. IV (B): kontrola wycieku rytmu."""
    rows = _load("exp_enron_temporal_split.csv")
    by = {}
    for r in rows:
        by[(r["mode"], r["metric"])] = float(r["mean"])
    lines = []
    for mode, lab in [
        ("leaky", "full history (\\emph{with leakage})"),
        ("split", "early window (\\emph{temporal split})"),
        ("split+oh", "early window $+\\,$off-hours (\\emph{both})"),
    ]:
        ta, tr = by.get((mode, "temporal_auc")), by.get((mode, "temporal_r1"))
        fa, fr = by.get((mode, "full_auc")), by.get((mode, "full_r1"))
        lines.append(f"{lab} & {_fmt(ta)} & {_fmt(tr)} & {_fmt(fa)} & {_fmt(fr)} \\\\")
    (OUT / "tab_enronsplit.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_enronsplit.tex")


def labelnoise():
    """Tab. S5a (A): szum etykiet."""
    rows = _load("cb_labelnoise.csv")
    names = [
        "COMPA",
        "hopper",
        "hopper-bez-precursor",
        "reczny-kontekst",
        "temporalny-GNN",
    ]
    lines = []
    for r in rows:
        vals = [float(r[n]) for n in names]
        mx = max(vals)
        cells = " & ".join(_fmt(v, v == mx) for v in vals)
        lines.append(f"{r['szum']}\\% & {cells} \\\\")
    (OUT / "tab_labelnoise.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_labelnoise.tex")


def sweep():
    """Tab. S2 (A): sweep p_infect."""
    rows = _load("exp_cascade_sweep.csv")
    vals = sorted({r["value"] for r in rows if r["sweep"] == "p_infect"})
    by = {(r["sweep"], r["value"], r["model"]): r for r in rows}
    lines = []
    for v in vals:
        cells = []
        for m in ["tab_1hop", "tab_ctx", "compa", "gnn_temporal"]:
            r = by.get(("p_infect", v, m))
            cells.append(_fmt(r["auc"]) if r else "---")
        lines.append(f"${v}$ & " + " & ".join(cells) + " \\\\")
    (OUT / "tab_sweep.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_sweep.tex")


def multiplex():
    """Tabela fabrykacji (B, B3): krzywa 0-100%."""
    rows = _load("exp_multiplex.csv")
    lines = []
    for r in rows:
        if not r["config"].startswith("fab") and r["config"] != "default":
            continue
        fab = r["fabrication"]
        lines.append(
            f"{int(float(fab) * 100)}\\% & {_fmt(r['mux_auc'])} & {_fmt(r['shuf_auc'])} & "
            f"{_fmt(r['fp_newsender_mux'])} \\\\"
        )
    (OUT / "tab_fabrication.tex").write_text("\n".join(lines), encoding="utf-8")
    print("ok: tab_fabrication.tex")


if __name__ == "__main__":
    for fn in (
        detectors,
        stealth,
        baselines,
        realgraphs,
        enronsplit,
        labelnoise,
        sweep,
        multiplex,
    ):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {fn.__name__}: {e}")
