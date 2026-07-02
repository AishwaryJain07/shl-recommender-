"""Phase 0: download + normalize the SHL catalog and print stats (BUILD_SPEC §11).

Usage:
    python scripts/fetch_catalog.py            # download, normalize, print stats
    python scripts/fetch_catalog.py --offline  # normalize an existing raw file

Writes:
    data/shl_catalog_raw.json         (raw download, gitignored — re-fetchable)
    data/catalog_normalized.json      (strictly-valid, bundled for deploy)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Make the repo root importable when run as a script (python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import catalog as cat  # noqa: E402
from app.config import CONFIG  # noqa: E402


def download(url: str, dest: Path, timeout: float) -> None:
    """Download the catalog to `dest`. Raises on failure (caller decides)."""
    print(f"Downloading catalog:\n  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "shl-recommender/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"  wrote {len(data):,} bytes -> {dest}")


def print_stats(items: list[dict]) -> None:
    """Print the Phase 0 catalog stats required by BUILD_SPEC §2/§11."""
    s = cat.compute_stats(items)
    print("=" * 60)
    print("CATALOG STATS")
    print("=" * 60)
    print(f"Total items: {s['total_items']}")

    print("\ntest_type distribution (items containing each code):")
    # Stable, readable order across the eight known codes.
    order = ["A", "B", "C", "D", "E", "K", "P", "S"]
    counts = s["code_counts"]
    for code in order:
        if code in counts:
            print(f"  {code}: {counts[code]}")
    # Surface any unexpected codes not in the canonical eight.
    for code, n in counts.items():
        if code not in order:
            print(f"  {code} (UNEXPECTED): {n}")

    print(f"\nMulti-key items (test_type like 'K,S'): {s['multi_key_items']}")
    print(f"Single-key items:                       {s['single_key_items']}")
    print(f"Distinct test_type combinations:        {s['distinct_test_type_combos']}")
    print("Top test_type combinations:")
    for combo, n in s["top_test_type_combos"]:
        print(f"  {combo:<12} {n}")

    print("\nData-quality checks (all should be 0):")
    print(f"  empty descriptions:  {s['empty_description']}")
    print(f"  missing URLs:        {s['missing_url']}")
    print(f"  missing entity_id:   {s['missing_entity_id']}")
    print(f"  missing name:        {s['missing_name']}")
    print(f"  missing test_type:   {s['missing_test_type']}")
    if s["unknown_keys"]:
        print(f"  UNKNOWN keys categories (catalog drift!): {s['unknown_keys']}")
    else:
        print("  unknown keys categories: none")
    print("=" * 60)


def main() -> int:
    # Force UTF-8 stdout so item names print cleanly on Windows (cp1252) consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — non-fatal; only affects console display
        pass

    ap = argparse.ArgumentParser(description="Fetch + normalize the SHL catalog.")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="skip download; normalize the existing raw file",
    )
    args = ap.parse_args()

    raw_path = CONFIG.raw_catalog_path
    norm_path = CONFIG.normalized_catalog_path

    if args.offline:
        if not raw_path.exists():
            print(f"ERROR: --offline but no raw file at {raw_path}", file=sys.stderr)
            return 1
        print(f"Offline mode: using cached {raw_path}")
    else:
        try:
            download(CONFIG.catalog_url, raw_path, CONFIG.http_timeout_s)
        except Exception as e:  # noqa: BLE001 — report and fall back honestly
            # Never fake success (BUILD_SPEC §10): fall back to cache if present,
            # otherwise stop with a clear error.
            if raw_path.exists():
                print(f"WARNING: download failed ({e!r}); using cached {raw_path}")
            else:
                print(
                    f"ERROR: download failed and no cached raw file at {raw_path}: {e!r}",
                    file=sys.stderr,
                )
                return 1

    # Load (lenient) -> normalize -> write strictly-valid JSON.
    raw_items = cat.load_raw(raw_path)
    items = cat.normalize_catalog(raw_items)
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    norm_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Normalized {len(items)} items -> {norm_path}\n")

    print_stats(items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
