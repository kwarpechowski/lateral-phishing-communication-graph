# -*- coding: utf-8 -*-
"""R4: scala czesciowe pliki lane'ow (enron per-rezym, realgraphs per-dataset)
do kanonicznych CSV po zakonczeniu fali. Uruchomic RAZ po fali."""

import csv
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _merge(patterns, out_name, header_from_first=True):
    """patterns: lista nazw plikow czesciowych; out: kanoniczny."""
    rows = []
    header = None
    for p in patterns:
        f = RESULTS / p
        if not f.exists():
            print(f"  [brak] {p}")
            continue
        with f.open(encoding="utf-8") as fh:
            r = list(csv.reader(fh))
            if not r:
                continue
            if header is None:
                header = r[0]
            rows.extend(r[1:])
    if header is None:
        print(f"  [skip] {out_name} (brak czesci)")
        return
    out = RESULTS / out_name
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  [ok] {out_name}: {len(rows)} wierszy")


def main():
    print("[merge-r4] Enron:")
    _merge(
        ["exp_cascade_enron_blast.csv", "exp_cascade_enron_stealthy.csv"],
        "exp_cascade_enron.csv",
    )
    _merge(
        ["exp_cascade_enron_timing_blast.csv", "exp_cascade_enron_timing_stealthy.csv"],
        "exp_cascade_enron_timing.csv",
    )
    _merge(
        [
            "exp_cascade_enron_blast_perseed.csv",
            "exp_cascade_enron_stealthy_perseed.csv",
        ],
        "exp_cascade_enron_perseed.csv",
    )
    print("[merge-r4] Realgraphs:")
    _merge(
        [
            "exp_cascade_realgraphs_email-Eu-core.csv",
            "exp_cascade_realgraphs_CollegeMsg.csv",
        ],
        "exp_cascade_realgraphs.csv",
    )
    _merge(
        [
            "exp_cascade_realgraphs_timing_email-Eu-core.csv",
            "exp_cascade_realgraphs_timing_CollegeMsg.csv",
        ],
        "exp_cascade_realgraphs_timing.csv",
    )
    _merge(
        [
            "exp_cascade_realgraphs_email-Eu-core_perseed.csv",
            "exp_cascade_realgraphs_CollegeMsg_perseed.csv",
        ],
        "exp_cascade_realgraphs_perseed.csv",
    )


if __name__ == "__main__":
    main()
