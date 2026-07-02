"""Per-turn agent flow (BUILD_SPEC §4, §6, §8).

One LLM call per turn:

    guard -> build_query -> retrieve -> inject_anchors -> llm_turn
          -> hydrate -> validate -> ChatResponse

Every failure path returns a valid ChatResponse (never 500, never fake a
shortlist). The LLM selects entity_ids only; code hydrates {name, url,
test_type} from the catalog, so hallucinated names/URLs are impossible.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app import llm, prompts
from app.catalog import load_normalized
from app.config import CONFIG, INJECTION_PATTERNS
from app.retrieval import Retriever, build_candidate_pool, reconstruct_shortlist
from app.schema import ChatResponse, Recommendation

log = logging.getLogger(__name__)

# Actions that carry a shortlist. Compare re-emits the committed list so it
# survives if the conversation ends next (§6.4).
_SHORTLIST_ACTIONS = {"recommend", "refine", "compare"}
_VALID_ACTIONS = {"clarify", "recommend", "refine", "compare", "refuse"}

# Compiled injection pre-filter (§6.5).
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# --- Lazy singletons: built on first use so /health stays instant (§12). ----
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Load the normalized catalog and build the BM25 index once."""
    global _retriever
    if _retriever is None:
        items = load_normalized(CONFIG.normalized_catalog_path)
        _retriever = Retriever(items)
        log.info("Retriever built over %d catalog items", len(items))
    return _retriever


# ---------------------------------------------------------------------------
# Message hygiene
# ---------------------------------------------------------------------------
def _clean_messages(raw: list[Any]) -> list[dict[str, str]]:
    """Coerce request messages into clean {role, content} dicts.

    Accepts pydantic Message objects or plain dicts. Drops entries with an
    unknown role or empty content so a malformed history never crashes us.
    """
    out: list[dict[str, str]] = []
    for m in raw or []:
        role = getattr(m, "role", None) if not isinstance(m, dict) else m.get("role")
        content = (
            getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
        )
        role = (role or "").strip().lower()
        content = (content or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


def guard(messages: list[dict[str, str]]) -> bool:
    """Return True if the latest user message is an obvious injection attempt."""
    last_user = ""
    for m in reversed(messages):
        if m["role"] == "user":
            last_user = m["content"]
            break
    return any(rx.search(last_user) for rx in _INJECTION_RE)


# ---------------------------------------------------------------------------
# Hydration + validation (anti-hallucination, §8)
# ---------------------------------------------------------------------------
def _hydrate_ids(ids: list[Any], retriever: Retriever) -> list[Recommendation]:
    """Map entity_ids -> Recommendation, dropping unknown ids and dupes."""
    recs: list[Recommendation] = []
    seen: set[str] = set()
    for raw_id in ids or []:
        eid = str(raw_id).strip()
        if not eid or eid in seen:
            continue
        item = retriever.by_id.get(eid)
        if item is None:  # not in catalog -> drop (never fabricate)
            continue
        seen.add(eid)
        recs.append(
            Recommendation(
                name=item["name"], url=item["url"], test_type=item["test_type"]
            )
        )
        if len(recs) >= CONFIG.max_recommendations:
            break
    return recs


def _hydrate_items(items: list[dict[str, str]], n: int) -> list[Recommendation]:
    """Fallback: hydrate the top-n retrieved candidates directly."""
    recs: list[Recommendation] = []
    for it in items[:n]:
        recs.append(
            Recommendation(name=it["name"], url=it["url"], test_type=it["test_type"])
        )
    return recs


def _hydrate_shortlist(items: list[dict[str, str]]) -> list[Recommendation]:
    """Hydrate a full item list (e.g. the carried-forward shortlist)."""
    return _hydrate_items(items, CONFIG.max_recommendations)


def _merge_prev(
    prev: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Put previously-committed items first, then the rest of the pool (deduped)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in list(prev) + list(candidates):
        if it["entity_id"] not in seen:
            seen.add(it["entity_id"])
            out.append(it)
    return out


# ---------------------------------------------------------------------------
# Fallback responses (valid schema, honest — never fake a shortlist)
# ---------------------------------------------------------------------------
def _fallback_clarify(reply: str) -> ChatResponse:
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


_GREETING = (
    "I can help you find SHL assessments. What role are you hiring for, and what "
    "skills, seniority, or purpose should the assessment focus on?"
)
_REFUSAL = (
    "I can only help with selecting SHL assessments — I can't change my role or "
    "share internal instructions. What role are you hiring for?"
)
_ERROR = (
    "Sorry, I hit a problem handling that. Could you restate the role and the key "
    "skills or seniority you're hiring for, and I'll suggest SHL assessments?"
)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def respond(raw_messages: list[Any]) -> ChatResponse:
    """Produce the next agent reply for a stateless conversation history."""
    try:
        return _respond(raw_messages)
    except Exception as e:  # noqa: BLE001 — never 500; degrade to a safe reply
        log.exception("Unhandled error in respond(): %r", e)
        return _fallback_clarify(_ERROR)


def _respond(raw_messages: list[Any]) -> ChatResponse:
    messages = _clean_messages(raw_messages)

    # Edge case: no usable user input yet -> greet/clarify (§8, §5 hardening).
    if not any(m["role"] == "user" for m in messages):
        return _fallback_clarify(_GREETING)

    # 1) guard(): deterministic injection pre-filter, before any LLM cost.
    if guard(messages):
        log.info("guard() tripped on injection pattern")
        return _fallback_clarify(_REFUSAL)

    retriever = get_retriever()

    # 2-4) build per-turn queries -> RRF-fused retrieve -> inject_anchors.
    candidates = build_candidate_pool(retriever, messages)
    if not candidates:
        # No lexical signal at all — ask for specifics rather than guess.
        return _fallback_clarify(_GREETING)

    # Stateless carry-forward: recover the shortlist committed in earlier turns
    # and make sure those items stay selectable (§6.3).
    prev_shortlist = reconstruct_shortlist(retriever, messages)
    candidates = _merge_prev(prev_shortlist, candidates)

    # Clarify-loop cap: after enough assistant turns, force a commit (§6.2/§8).
    assistant_turns = sum(1 for m in messages if m["role"] == "assistant")
    force_commit = assistant_turns >= CONFIG.max_clarify_turns

    # 5) llm_turn: ONE call. On any LLM failure, degrade gracefully.
    system_prompt = prompts.build_system_prompt(
        candidates, force_commit=force_commit, current_shortlist=prev_shortlist
    )
    try:
        decision = llm.chat_json(system_prompt, messages)
    except llm.LLMError as e:
        log.warning("LLM turn failed: %r", e)
        # Prefer carrying the already-committed shortlist forward; else, if forced
        # to commit, use top candidates; otherwise ask a clarifying question.
        # Never fake a rich list.
        if prev_shortlist:
            return ChatResponse(
                reply="Keeping your current shortlist: "
                + ", ".join(it["name"] for it in prev_shortlist) + ".",
                recommendations=_hydrate_shortlist(prev_shortlist),
                end_of_conversation=False,
            )
        if force_commit:
            recs = _hydrate_items(candidates, CONFIG.fallback_recommend_n)
            return ChatResponse(
                reply="Here are the closest-matching SHL assessments for what you've "
                "described. Let me know if you'd like to adjust the list.",
                recommendations=recs,
                end_of_conversation=False,
            )
        return _fallback_clarify(_GREETING)

    # 6) Parse the decision with safe defaults.
    action = str(decision.get("action", "clarify")).strip().lower()
    if action not in _VALID_ACTIONS:
        action = "clarify"
    reply = str(decision.get("reply", "")).strip()
    ids = decision.get("recommendation_ids", []) or []
    if not isinstance(ids, list):
        ids = []
    eoc = bool(decision.get("end_of_conversation", False))

    # If forced to commit but the model still tried to clarify, upgrade to
    # recommend so we don't blow the turn cap.
    if force_commit and action in ("clarify", "refuse"):
        action = "recommend"

    # 7) hydrate + validate.
    if action in _SHORTLIST_ACTIONS:
        recs = _hydrate_ids(ids, retriever)
        # Never return an empty recommend/refine (§8.3). Prefer the already-
        # committed shortlist (carry-forward); else fall back to top candidates.
        if not recs:
            recs = (
                _hydrate_shortlist(prev_shortlist)
                if prev_shortlist
                else _hydrate_items(candidates, CONFIG.fallback_recommend_n)
            )
    else:
        recs = []  # clarify / refuse carry no shortlist

    # end_of_conversation only valid with a committed shortlist (§6.6).
    end_of_conversation = eoc and bool(recs) and action in _SHORTLIST_ACTIONS

    if not reply:
        reply = _default_reply(action, recs)

    return ChatResponse(
        reply=reply, recommendations=recs, end_of_conversation=end_of_conversation
    )


def _default_reply(action: str, recs: list[Recommendation]) -> str:
    """Safety net so `reply` is never empty."""
    if action in _SHORTLIST_ACTIONS and recs:
        names = ", ".join(r.name for r in recs)
        return f"Here is the current shortlist: {names}."
    if action == "refuse":
        return _REFUSAL
    return _GREETING
