"""Phase 0: parse the 10 public conversation traces (BUILD_SPEC §9, §11).

For each trace we extract:
  * user_turns    — the ordered list of user messages (replayed during eval).
  * gold_shortlist — the FINAL markdown table in the file, as [{name, url}].

We use the LAST table in the file as the labelled shortlist (NOT turn numbers),
because trace numbering is inconsistent — e.g. C10 skips "Turn 3". The final
table is the committed shortlist the harness scores Recall@10 against.

If a normalized catalog is present, each gold item is checked against it (by URL,
then by name) so we can confirm the labels are in-catalog — i.e. that Recall@10
has a 100% ceiling and the gold URLs are real.

Writes eval/traces_parsed.json and prints a summary.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# Make the repo root importable when run as a script (python eval/...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import catalog as cat  # noqa: E402
from app.config import CONFIG  # noqa: E402

# A user message begins after a "**User**" marker; the agent's begins after
# "**Agent**". Blockquote lines ("> ...") carry the message body.
_USER_MARKER = re.compile(r"^\*\*User\*\*\s*$")
_AGENT_MARKER = re.compile(r"^\*\*Agent\*\*\s*$")
# Extract a URL from a table cell, whether wrapped as <url> or [text](url).
_MD_LINK = re.compile(r"\[[^\]]*\]\((?P<url>[^)]+)\)")


def _trace_sort_key(p: Path) -> tuple[int, str]:
    """Natural sort so C10 comes after C9 (numeric part of the stem)."""
    m = re.search(r"(\d+)", p.stem)
    return (int(m.group(1)) if m else 0, p.stem)


def extract_user_turns(text: str) -> list[str]:
    """Return the ordered user messages from a trace's markdown."""
    lines = text.splitlines()
    turns: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if _USER_MARKER.match(lines[i].strip()):
            i += 1
            # Skip blank lines between the marker and the quoted message.
            while i < n and lines[i].strip() == "":
                i += 1
            # Collect consecutive blockquote lines (the user's message body),
            # allowing blank ">" lines inside a multi-line quote (e.g. the C9 JD).
            body: list[str] = []
            while i < n:
                stripped = lines[i].strip()
                if stripped.startswith(">"):
                    body.append(stripped[1:].lstrip())
                    i += 1
                elif stripped == "" and body:
                    # A blank separator — peek: continue only if more quote follows.
                    j = i + 1
                    while j < n and lines[j].strip() == "":
                        j += 1
                    if j < n and lines[j].strip().startswith(">"):
                        body.append("")
                        i = j
                    else:
                        break
                else:
                    break
            msg = "\n".join(body).strip()
            if msg:
                turns.append(msg)
        else:
            i += 1
    return turns


def _split_table_row(line: str) -> list[str]:
    """Split a markdown table row into trimmed cells."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    """A markdown header/body separator row is all dashes/colons."""
    return bool(cells) and all(set(c) <= set("-: ") and "-" in c for c in cells)


def extract_tables(text: str) -> list[list[list[str]]]:
    """Return every markdown table as a list of rows (each row = list of cells)."""
    tables: list[list[list[str]]] = []
    block: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            block.append(line)
        else:
            if block:
                tables.append([_split_table_row(l) for l in block])
                block = []
    if block:
        tables.append([_split_table_row(l) for l in block])
    # Keep only real tables: header + separator + >=1 data row.
    return [t for t in tables if len(t) >= 3 and _is_separator_row(t[1])]


def _clean_url(cell: str) -> str:
    """Extract a clean URL from a table cell (<url> or [text](url) or bare)."""
    cell = cell.strip()
    m = _MD_LINK.search(cell)
    if m:
        return m.group("url").strip()
    return cell.strip("<> ").strip()


def parse_gold_shortlist(text: str) -> list[dict[str, str]]:
    """Parse the LAST markdown table into [{name, url}]."""
    tables = extract_tables(text)
    if not tables:
        return []
    table = tables[-1]
    header = [h.lower() for h in table[0]]
    # Locate the Name and URL columns robustly (order/column-count may vary).
    name_idx = header.index("name") if "name" in header else 1
    url_idx = header.index("url") if "url" in header else len(header) - 1

    rows: list[dict[str, str]] = []
    for cells in table[2:]:  # skip header + separator
        if len(cells) <= max(name_idx, url_idx):
            continue
        name = cells[name_idx].strip()
        url = _clean_url(cells[url_idx])
        if name:
            rows.append({"name": name, "url": url})
    return rows


def _norm_url(u: str) -> str:
    return u.strip().rstrip("/").lower()


def _norm_name(nm: str) -> str:
    return re.sub(r"\s+", " ", nm).strip().lower()


def _build_catalog_lookups() -> tuple[set[str], set[str]] | None:
    """Return (url_set, name_set) from the normalized catalog, or None if absent."""
    path = CONFIG.normalized_catalog_path
    if not path.exists():
        return None
    items = cat.load_normalized(path)
    urls = {_norm_url(it["url"]) for it in items}
    names = {_norm_name(it["name"]) for it in items}
    return urls, names


def main() -> int:
    # Trace names contain en/em-dashes; force UTF-8 stdout so the summary prints
    # cleanly on Windows consoles (cp1252 by default). Data is always UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — non-fatal; only affects console display
        pass

    trace_files = sorted(CONFIG.traces_dir.glob("*.md"), key=_trace_sort_key)
    if not trace_files:
        print(f"ERROR: no trace files found in {CONFIG.traces_dir}", file=sys.stderr)
        return 1

    lookups = _build_catalog_lookups()
    if lookups is None:
        print(
            "NOTE: normalized catalog not found — skipping in-catalog check.\n"
            "      Run scripts/fetch_catalog.py first for the FOUND/MISSING column.\n"
        )

    parsed: list[dict[str, Any]] = []
    total_gold = 0
    total_missing = 0

    for f in trace_files:
        text = f.read_text(encoding="utf-8")
        user_turns = extract_user_turns(text)
        gold = parse_gold_shortlist(text)
        total_gold += len(gold)

        # Annotate each gold item with in-catalog match status.
        if lookups is not None:
            url_set, name_set = lookups
            for g in gold:
                in_cat = (
                    _norm_url(g["url"]) in url_set or _norm_name(g["name"]) in name_set
                )
                g["in_catalog"] = in_cat
                if not in_cat:
                    total_missing += 1

        parsed.append(
            {
                "trace_id": f.stem,
                "file": str(f.relative_to(CONFIG.traces_dir.parent.parent)),
                "user_turns": user_turns,
                "gold_shortlist": gold,
                "gold_count": len(gold),
            }
        )

    # --- Print summary ---
    print("=" * 72)
    print("PARSED TRACES — gold shortlists (last table per file)")
    print("=" * 72)
    for p in parsed:
        print(f"\n{p['trace_id']}  ({p['gold_count']} gold items, "
              f"{len(p['user_turns'])} user turns)")
        for i, turn in enumerate(p["user_turns"], 1):
            preview = turn.replace("\n", " ")
            if len(preview) > 90:
                preview = preview[:87] + "..."
            print(f"    U{i}: {preview}")
        print("    Gold shortlist:")
        for g in p["gold_shortlist"]:
            mark = ""
            if "in_catalog" in g:
                mark = "  [FOUND]" if g["in_catalog"] else "  [** MISSING **]"
            print(f"      - {g['name']}{mark}")
            print(f"          {g['url']}")

    print("\n" + "=" * 72)
    print(f"Traces parsed: {len(parsed)} | total gold items: {total_gold}", end="")
    if lookups is not None:
        print(f" | not-in-catalog: {total_missing}")
    else:
        print("")
    print("=" * 72)

    # --- Persist for the eval harness (Phase 4) ---
    out = CONFIG.traces_parsed_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
