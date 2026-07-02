"""Catalog loading, normalization, lookup, and stats (BUILD_SPEC §2, §4, §8).

Responsibilities:
  * Read the raw SHL product-catalog JSON. The source contains unescaped control
    characters inside some `description` fields, so it is parsed with
    `strict=False`; the normalized output we write back is strictly valid JSON.
  * Normalize every item into a clean, uniform record.
  * Derive the single-letter `test_type` from `keys[]` IN CODE — the LLM never
    authors `test_type` or URLs (anti-hallucination, BUILD_SPEC §3, §8).
  * Provide an entity_id -> item lookup used later to hydrate recommendations.
  * Compute catalog statistics for Phase 0 verification.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import KEYS_TO_TEST_TYPE

# Collapse any run of whitespace (incl. the stray newlines/tabs in the source)
# into a single space when building retrieval text.
_WHITESPACE_RE = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    """Normalize whitespace in free text."""
    return _WHITESPACE_RE.sub(" ", (text or "")).strip()


def keys_to_test_type(keys: list[str]) -> tuple[str, list[str]]:
    """Map catalog `keys` category names to a joined single-letter code string.

    Returns (test_type, unknown_keys). Code order follows the item's own `keys`
    order (this matches the sample traces, e.g. ['Personality & Behavior',
    'Competencies'] -> 'P,C'); duplicate codes are removed. Any category not in
    the mapping is skipped and reported back so callers can flag catalog drift.
    """
    codes: list[str] = []
    unknown: list[str] = []
    for k in keys or []:
        code = KEYS_TO_TEST_TYPE.get(k)
        if code is None:
            unknown.append(k)
        elif code not in codes:
            codes.append(code)
    return ",".join(codes), unknown


def normalize_item(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn one raw catalog row into a normalized record.

    Keeps every field a downstream stage might need. `url` comes only from the
    source `link`, and `test_type` is derived here — neither is ever produced by
    the model.
    """
    keys = raw.get("keys") or []
    name = _clean(raw.get("name"))
    description = raw.get("description") or ""
    job_levels = raw.get("job_levels") or []
    test_type, _unknown = keys_to_test_type(keys)

    # Per-item retrieval document (BUILD_SPEC §5): name + description + keys +
    # job_levels, whitespace-normalized. Built here; indexed in Phase 1.
    # TODO(Phase 1): retrieval.py tokenizes/indexes `search_text`.
    search_text = _clean(
        " ".join([name, _clean(description), " ".join(keys), " ".join(job_levels)])
    )

    return {
        "entity_id": str(raw.get("entity_id", "")).strip(),
        "name": name,
        "url": _clean(raw.get("link")),          # `link` is the ONLY URL source
        "test_type": test_type,                   # derived in code, never by LLM
        "keys": keys,
        "description": description,
        "job_levels": job_levels,
        "languages": raw.get("languages") or [],
        "duration": raw.get("duration") or "",
        "remote": raw.get("remote") or "",
        "adaptive": raw.get("adaptive") or "",
        "status": raw.get("status") or "",
        "scraped_at": raw.get("scraped_at") or "",
        "search_text": search_text,
    }


def load_raw(path: Path) -> list[dict[str, Any]]:
    """Load the raw catalog JSON, tolerating unescaped control characters."""
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text, strict=False)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list at {path}, got {type(data).__name__}"
        )
    return data


def normalize_catalog(raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize every raw item."""
    return [normalize_item(it) for it in raw_items]


def build_id_index(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """entity_id -> normalized item. Used by the hydrator (Phase 2/3)."""
    return {it["entity_id"]: it for it in items if it.get("entity_id")}


def load_normalized(path: Path) -> list[dict[str, Any]]:
    """Load the pre-built, strictly-valid normalized catalog (app boot path)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compute_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Phase 0 verification stats over normalized items."""
    total = len(items)

    # test_type distribution: count how many items include each single-letter
    # code (a "K,S" item counts toward both K and S).
    code_counts: Counter[str] = Counter()
    combo_counts: Counter[str] = Counter()
    multi_key = 0
    missing_test_type = 0
    empty_description = 0
    missing_url = 0
    missing_id = 0
    missing_name = 0
    unknown_keys: Counter[str] = Counter()

    for it in items:
        tt = it["test_type"]
        combo_counts[tt or "(none)"] += 1
        codes = [c for c in tt.split(",") if c]
        for c in codes:
            code_counts[c] += 1
        if len(codes) > 1:
            multi_key += 1
        if not codes:
            missing_test_type += 1
        if not (it.get("description") or "").strip():
            empty_description += 1
        if not it.get("url"):
            missing_url += 1
        if not it.get("entity_id"):
            missing_id += 1
        if not it.get("name"):
            missing_name += 1
        # Re-check keys against the mapping to surface any catalog drift.
        _, unknown = keys_to_test_type(it.get("keys") or [])
        for u in unknown:
            unknown_keys[u] += 1

    return {
        "total_items": total,
        "code_counts": dict(code_counts.most_common()),
        "distinct_test_type_combos": len(combo_counts),
        "top_test_type_combos": combo_counts.most_common(10),
        "multi_key_items": multi_key,
        "single_key_items": total - multi_key - missing_test_type,
        "missing_test_type": missing_test_type,
        "empty_description": empty_description,
        "missing_url": missing_url,
        "missing_entity_id": missing_id,
        "missing_name": missing_name,
        "unknown_keys": dict(unknown_keys),
    }
