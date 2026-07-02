"""Retrieval diagnostic (BUILD_SPEC §5, Phase 1 "done when").

For each trace, builds the query from all user turns, retrieves top-K + anchors,
and reports what fraction of the gold items land in the candidate pool. This is
the RECALL CEILING: the LLM can only recommend items that reach the pool. Needs
no LLM/server — pure retrieval.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import catalog as cat  # noqa: E402
from app.config import CONFIG  # noqa: E402
from app.retrieval import Retriever, build_candidate_pool  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass


def _norm_url(u: str) -> str:
    return (u or "").strip().rstrip("/").lower()


def main() -> int:
    items = cat.load_normalized(CONFIG.normalized_catalog_path)
    retriever = Retriever(items)
    url_to_id = {_norm_url(it["url"]): it["entity_id"] for it in items}

    traces = json.loads(CONFIG.traces_parsed_path.read_text(encoding="utf-8"))

    print("=" * 72)
    print(f"CANDIDATE-POOL RECALL  (pool_size={CONFIG.retrieval_pool_size}, "
          f"per_query_k={CONFIG.retrieval_per_query_k}, RRF + "
          f"{len(CONFIG.anchor_names)} anchors)")
    print("=" * 72)
    total_recall = 0.0
    for t in traces:
        messages = [{"role": "user", "content": u} for u in t["user_turns"]]
        pool = build_candidate_pool(retriever, messages)
        pool_ids = {it["entity_id"] for it in pool}

        gold = t["gold_shortlist"]
        missing = []
        hits = 0
        for g in gold:
            gid = url_to_id.get(_norm_url(g["url"]))
            if gid in pool_ids:
                hits += 1
            else:
                missing.append(g["name"])
        recall = hits / len(gold) if gold else 1.0
        total_recall += recall
        flag = "" if recall == 1.0 else "  <-- GAP"
        print(f"{t['trace_id']:<5} pool={len(pool):>3}  gold={len(gold)}  "
              f"in_pool={hits}  recall={recall:.2f}{flag}")
        for m in missing:
            print(f"        MISSING: {m}")
    print("-" * 72)
    print(f"MEAN candidate-pool recall: {total_recall / len(traces):.3f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
