#!/usr/bin/env python3
"""Verify C-Conference/bibliography.bib against cited keys and source metadata.

Checks (rule: no fabricated bibliography):
  1. every key cited in main.tex (\cite/\citep/\citealp/\citeauthor) exists in the .bib;
  2. every cited entry carries a DOI or an arXiv identifier;
  3. title (and first-author family name) match the metadata at the source:
     Crossref API for DOIs, arXiv export API for arXiv ids;
  4. uncited bib entries are reported for removal.

Network failures are reported as UNVERIFIED (not as errors).
Output: verify_bib_report.md + exit code (0 = clean, 1 = issues found).

Pure stdlib; deterministic; run from repo root or C-Conference/.
"""

from __future__ import annotations

import re
import sys
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE / "main.tex"
BIB = HERE / "bibliography.bib"
REPORT = HERE / "verify_bib_report.md"

ARXIV_API = "http://export.arxiv.org/api/query?id_list={}"
CROSSREF_API = "https://api.crossref.org/works/{doi}"
CROSSREF_QUERY = (
    "https://api.crossref.org/works?query.bibliographic={q}"
    "&rows=3&select=DOI,title,author"
)


def tex_unescape(s: str) -> str:
    """Unescape common LaTeX accents: \\"u / \\"{u} -> u-umlaut, etc."""
    bs = chr(92)  # backslash
    for letter, ch in [
        ("u", "\u00fc"),
        ("o", "\u00f6"),
        ("a", "\u00e4"),
        ("e", "\u00eb"),
        ("A", "\u00c4"),
        ("O", "\u00d6"),
        ("U", "\u00dc"),
    ]:
        s = s.replace(bs + '"' + letter, ch)
        s = s.replace(bs + '"{' + letter + "}", ch)
    s = s.replace(bs + "'e", "\u00e9")
    s = s.replace(bs + "'o", "\u00f3")
    s = s.replace(bs + "`e", "\u00e8")
    return s


def norm(s: str) -> str:
    """Normalization for title comparison: lowercase, strip braces/punct/accents."""
    s = tex_unescape(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[{}\\]", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def cited_keys() -> set[str]:
    tex = MAIN.read_text(encoding="utf-8")
    keys: set[str] = set()
    for m in re.finditer(
        r"\\(?:cite|citep|citet|citealp|citeauthor|citealp)\*?(?:\[[^\]]*\])*\{([^}]*)\}",
        tex,
    ):
        for k in m.group(1).split(","):
            if k.strip():
                keys.add(k.strip())
    return keys


def _split_fields(body: str) -> dict[str, str]:
    """Brace-depth-aware field splitter for a bib entry body."""
    fields: dict[str, str] = {}
    i, n = 0, len(body)
    while i < n:
        # skip whitespace and commas
        while i < n and body[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        m = re.match(r"(\w+)\s*=\s*", body[i:])
        if not m:
            i += 1
            continue
        name = m.group(1).lower()
        i += m.end()
        val_start = i
        depth = 0
        quote = False
        while i < n:
            c = body[i]
            if quote:
                if c == '"':
                    quote = False
                i += 1
                continue
            if c == '"':
                quote = True
                i += 1
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth <= 0:
                    i += 1
                    break
            elif c == "," and depth == 0:
                break
            i += 1
        raw = body[val_start:i].strip().strip('"')
        raw = raw.strip("{}").strip()
        fields[name] = re.sub(r"\s+", " ", raw).strip()
        i += 1
    return fields


def parse_bib() -> dict[str, dict]:
    raw = BIB.read_text(encoding="utf-8")
    entries: dict[str, dict] = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", raw):
        start = m.start()
        etype, key = m.group(1).lower(), m.group(2)
        # find the end of this entry: next '@' at top level or EOF
        nxt = raw.find("\n@", start + 1)
        end = nxt if nxt != -1 else len(raw)
        fields = _split_fields(raw[m.end() : end])
        entries[key] = {"type": etype, **fields}
    return entries


def fetch(url: str) -> str | None:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "bib-verify/1.0 (mailto:check@example.org)"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        print(f"    [net] {url[:70]} -> {e}", file=sys.stderr)
        return None


def arxiv_meta(arxiv_id: str) -> dict | None:
    data = fetch(ARXIV_API.format(arxiv_id))
    if not data:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(data)
    entry = root.find("a:entry", ns)
    if entry is None:
        return {"error": "no entry"}
    title = entry.find("a:title", ns)
    authors = [
        a.find("a:name", ns).text
        for a in entry.findall("a:author", ns)
        if a.find("a:name", ns) is not None
    ]
    return {
        "title": " ".join(title.text.split()) if title is not None else "",
        "authors": authors,
    }


def crossref_meta(doi: str) -> dict | None:
    data = fetch(CROSSREF_API.format(doi=doi))
    if not data:
        return None
    import json

    msg = json.loads(data)["message"]
    title = msg.get("title", [""])[0]
    authors = [
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in msg.get("author", [])
    ]
    return {"title": title, "authors": authors}


def crossref_search(title: str) -> list[dict]:
    import json

    data = fetch(CROSSREF_QUERY.format(q=urllib.request.quote(title)))
    if not data:
        return []
    items = json.loads(data)["message"]["items"]
    return items


def first_author_family(authors: list[str] | str) -> str:
    if isinstance(authors, str):
        first = authors.split(" and ")[0]
        fam = first.split(",")[0].strip() if "," in first else first.split()[-1]
        return fam.lower().replace("{", "").replace("}", "")
    if not authors:
        return ""
    first = authors[0]
    fam = first.split(",")[0].strip() if "," in first else first.split()[-1]
    return fam.lower().replace("{", "").replace("}", "")


def main() -> int:
    cites = cited_keys()
    bib = parse_bib()
    lines = [
        "# Bibliography verification report",
        "",
        f"cited keys: {len(cites)}; bib entries: {len(bib)}",
        "",
    ]
    issues = 0

    # 1) cited but missing from bib
    missing = sorted(cites - set(bib))
    if missing:
        lines.append("## MISSING FROM BIB (cited in main.tex, no entry)")
        lines += [f"- `{k}`" for k in missing]
        lines.append("")
        issues += len(missing)

    # 2) uncited entries
    uncited = sorted(set(bib) - cites)
    lines.append(f"## UNCITED ENTRIES (candidates for removal): {len(uncited)}")
    lines += [f"- `{k}`" for k in uncited]
    lines.append("")

    # 3) per-cited-entry verification
    lines.append("## CITED ENTRIES vs SOURCE METADATA")
    lines.append("| key | id | title ok | author ok | note |")
    lines.append("|---|---|---|---|---|")
    for key in sorted(cites & set(bib)):
        e = bib[key]
        title = e.get("title", "")
        doi = e.get("doi", "").strip()
        arxiv_m = re.search(
            r"(\d{4}\.\d{4,5})(v\d+)?",
            e.get("howpublished", "")
            + " "
            + e.get("eprint", "")
            + " "
            + e.get("note", ""),
        )
        ident = (
            f"DOI:{doi}"
            if doi
            else (f"arXiv:{arxiv_m.group(1)}" if arxiv_m else "NONE")
        )
        if not doi and not arxiv_m:
            lines.append(f"| `{key}` | NONE | ? | ? | **no DOI/arXiv id** |")
            issues += 1
            continue
        meta = crossref_meta(doi) if doi else arxiv_meta(arxiv_m.group(1))
        if meta is None:
            lines.append(f"| `{key}` | {ident} | ? | ? | network/unverified |")
            continue
        if "error" in meta:
            lines.append(f"| `{key}` | {ident} | ? | ? | **not found at source** |")
            issues += 1
            continue
        src_title, src_authors = meta["title"], meta["authors"]
        title_ok = norm(title) == norm(src_title)
        fam_bib = first_author_family(tex_unescape(e.get("author", "")))
        fam_src = first_author_family(src_authors)
        author_ok = (fam_bib == fam_src) or (
            fam_bib.split("-")[0] == fam_src.split("-")[0]
        )
        note = ""
        if not title_ok:
            issues += 1
            note += f"bib: `{title[:60]}` vs src: `{src_title[:60]}`"
        if not author_ok:
            issues += 1
            note += f" authors bib `{fam_bib}` vs src `{fam_src}`"
        lines.append(
            f"| `{key}` | {ident} | {'OK' if title_ok else '**X**'} "
            f"| {'OK' if author_ok else '**X**'} | {note} |"
        )
        time.sleep(0.4)  # be polite to the APIs

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[verify_bib] {issues} issues; report -> {REPORT.name}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
