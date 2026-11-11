# -*- coding: utf-8 -*-
"""Czysci komentarze rewizyjne przed faktyczna submisja.

Usuwa z plikow .tex:
  * linie zaczynajace sie od '% ZMIANA' (wyjasnienia zmian po recenzjach —
    zostawione w repo do audytu, ale submisja ma je nie zawierac),
  * linie '% NOWE:' (ten sam charakter).

Uzycie: python strip_zmiana.py <plik.tex> [wiecej .tex ...]
Bez argumentow: czysci wszystkie .tex artykulow A-Journal i B-Conference IN-PLACE.
Zalecane: najpierw skopiuj pliki (backup), albo uzywaj na czystym checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [
    ROOT / "artykuly_bogdan" / "A-Journal" / "main.tex",
    ROOT / "artykuly_bogdan" / "A-Journal" / "supplementary.tex",
    *sorted((ROOT / "artykuly_bogdan" / "A-Journal" / "en").glob("*.tex")),
    ROOT / "artykuly_bogdan" / "B-Conference" / "main.tex",
]


def strip(path: Path) -> int:
    src = path.read_text(encoding="utf-8")
    keep = [
        ln
        for ln in src.splitlines()
        if not (ln.lstrip().startswith("% ZMIANA") or ln.lstrip().startswith("% NOWE:"))
    ]
    out = "\n".join(keep)
    if src.endswith("\n") and not out.endswith("\n"):
        out += "\n"
    removed = len(src.splitlines()) - len(keep)
    path.write_text(out, encoding="utf-8", newline="\n")
    return removed


def main() -> None:
    files = [Path(a) for a in sys.argv[1:]] or TARGETS
    for f in files:
        if not f.exists():
            print(f"  [skip] {f} (brak)")
            continue
        n = strip(f)
        print(f"  [ok] {f.name}: usunieto {n} linii komentarzy rewizyjnych")


if __name__ == "__main__":
    main()
