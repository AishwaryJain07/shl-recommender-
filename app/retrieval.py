"""Lexical retrieval over the catalog (BUILD_SPEC §5).

Phase 1 baseline: a BM25 index over each item's `search_text`
(name + description + keys + job_levels). The per-turn query is the concatenation
of all user-side messages (constraints accumulate across turns). We return the
top-K candidates and inject the recurring anchor instruments into the pool.

TODO(Phase 4b): optionally add dense retrieval + RRF fusion if the eval shows a
recall gap on the semantic traces.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from rank_bm25 import BM25Okapi

from app.config import CONFIG

# Tokenizer: lowercase alphanumeric runs. Simple, deterministic, language-neutral
# enough for this catalog (English names/descriptions).
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _doc_tokens(item: dict[str, Any]) -> list[str]:
    """Build the weighted BM25 document for one item (field boosting).

    Name and keys tokens are repeated so matches there outrank prose-only
    matches. Description and job_levels add recall for semantic-ish phrasing.
    """
    name = _tokenize(item.get("name", ""))
    keys = _tokenize(" ".join(item.get("keys", [])))
    desc = _tokenize(item.get("description", ""))
    jl = _tokenize(" ".join(item.get("job_levels", [])))
    return name * CONFIG.name_boost + keys * CONFIG.keys_boost + desc + jl


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "")).strip().lower()


class Retriever:
    """BM25 index + anchor injection over the normalized catalog."""

    def __init__(self, items: list[dict[str, Any]]):
        self.items = items
        self.by_id: dict[str, dict[str, Any]] = {
            it["entity_id"]: it for it in items if it.get("entity_id")
        }
        self._by_norm_name: dict[str, dict[str, Any]] = {
            _norm_name(it["name"]): it for it in items if it.get("name")
        }
        # Tokenized corpus, one field-weighted document per catalog item.
        self._corpus_tokens: list[list[str]] = [_doc_tokens(it) for it in items]
        self._bm25 = BM25Okapi(self._corpus_tokens)

        # Pre-resolve anchor names -> items once (config names are exact catalog
        # names, verified in Phase 0). Missing anchors are skipped, not faked.
        self._anchors: list[dict[str, Any]] = []
        for nm in CONFIG.anchor_names:
            item = self._by_norm_name.get(_norm_name(nm))
            if item is not None:
                self._anchors.append(item)

    def _ranked_indices(self, query: str, k: int) -> list[int]:
        """Indices of the top-k catalog items for one query (positive scores)."""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked[:k] if scores[i] > 0]

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return the top-K catalog items for a single query, by BM25 score."""
        k = top_k or CONFIG.retrieval_top_k
        return [self.items[i] for i in self._ranked_indices(query, k)]

    def retrieve_fused(
        self,
        queries: list[str],
        per_query_k: int | None = None,
        final_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fuse per-query BM25 rankings with Reciprocal Rank Fusion (RRF).

        RRF score for an item = sum over queries of 1 / (rrf_k + rank), where
        rank is 0-based within that query's top-`per_query_k`. Items surfaced by
        multiple sub-queries (or ranked highly by one) rise to the top. This
        recovers single-skill tests that a single concatenated query buries.
        """
        pq = per_query_k or CONFIG.retrieval_per_query_k
        fk = final_k or CONFIG.retrieval_pool_size
        agg: dict[int, float] = defaultdict(float)
        for q in queries:
            for rank, idx in enumerate(self._ranked_indices(q, pq)):
                agg[idx] += 1.0 / (CONFIG.rrf_k + rank)
        ordered = sorted(agg, key=lambda i: agg[i], reverse=True)[:fk]
        return [self.items[i] for i in ordered]

    def inject_anchors(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Append the configured anchor items not already in the candidate pool.

        Anchors join the POOL only; the policy decides whether to actually use
        them and must honor removals (BUILD_SPEC §5).
        """
        present = {it["entity_id"] for it in candidates}
        out = list(candidates)
        for anchor in self._anchors:
            if anchor["entity_id"] not in present:
                out.append(anchor)
                present.add(anchor["entity_id"])
        return out


def user_messages(messages: list[dict[str, str]]) -> list[str]:
    """Return the non-empty user-side message contents, in order."""
    return [
        m.get("content", "").strip()
        for m in messages
        if m.get("role") == "user" and m.get("content", "").strip()
    ]


def build_query(messages: list[dict[str, str]]) -> str:
    """Concatenate all user-side messages into one combined retrieval query."""
    return "\n".join(user_messages(messages))


def reconstruct_shortlist(
    retriever: Retriever, messages: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Recover the previously-committed shortlist from the last assistant reply.

    The API is stateless and only our `reply` text survives in the history (not
    the structured recommendations). Since the policy restates the shortlist by
    name every turn, we recover it by finding catalog names present in the last
    assistant message. Used to (a) guarantee those items stay in the candidate
    pool, (b) hint the LLM to carry them forward, and (c) provide a sane fallback
    on a confirmation turn — instead of regenerating from scratch.
    """
    last = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content", "").strip():
            last = m["content"]
            break
    if not last:
        return []
    text = last.lower()
    hits: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()
    for it in retriever.items:
        nm = it["name"].strip().lower()
        if len(nm) < 5:
            continue
        idx = text.find(nm)
        # Require a left boundary so "SQL (New)" doesn't match inside "PL/SQL (New)".
        if idx == -1:
            continue
        if idx > 0 and (text[idx - 1].isalnum() or text[idx - 1] in "/-"):
            continue
        if it["entity_id"] not in seen:
            seen.add(it["entity_id"])
            hits.append((idx, it))
    hits.sort(key=lambda x: x[0])
    return [it for _, it in hits]


def build_candidate_pool(
    retriever: Retriever, messages: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Construct the per-turn candidate pool (SINGLE source of truth).

    Queries = each individual user turn + the combined query (when there is more
    than one turn). Results are RRF-fused, then the anchor instruments are
    injected. Used by both the agent and the retrieval diagnostics so they never
    drift.
    """
    users = user_messages(messages)
    if not users:
        return []
    queries = list(users)
    if len(users) > 1:
        queries.append("\n".join(users))  # combined query captures cross-turn context
    pool = retriever.retrieve_fused(queries)
    return retriever.inject_anchors(pool)
