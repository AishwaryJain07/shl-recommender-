"""Central configuration (BUILD_SPEC §10: no hardcoding).

All tunables live here or in `.env`. Nothing operational is hardcoded inside
business logic. Values are read from environment variables (loaded from a local
`.env` if present) with safe defaults, so the service still boots when a
variable is unset (in production the host injects real values).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Repo root = parent of the app/ package, resolved so paths work from any CWD
# (scripts run from repo root, the app runs from wherever the host launches it).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env from the repo root if it exists. No-op in production, where env vars
# are injected by the host (BUILD_SPEC §12: keep secrets out of the repo).
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    """Read an env var, treating empty string as unset (falls back to default)."""
    val = os.getenv(name)
    return val if val not in (None, "") else default


def _path(name: str, default_rel: str) -> Path:
    """Resolve a path env var; relative values are anchored to PROJECT_ROOT."""
    raw = _env(name, default_rel)
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# ---------------------------------------------------------------------------
# keys[] -> single-letter test_type code (BUILD_SPEC §2).
#
# This is a FIXED domain mapping (not a per-deploy tunable), so it lives here as
# a module constant rather than in .env. `test_type` in every response is
# ALWAYS derived from this table in code — the LLM never authors it. The catalog
# uses exactly these eight category names (verified in Phase 0).
# ---------------------------------------------------------------------------
KEYS_TO_TEST_TYPE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# ---------------------------------------------------------------------------
# Deterministic prompt-injection pre-filter (BUILD_SPEC §6.5, guard()).
#
# A light first line of defense: obvious injection / role-hijack strings are
# refused BEFORE the LLM call. The policy prompt handles subtler cases. Kept
# here (not hardcoded in agent.py) so the patterns are tunable. Matched
# case-insensitively as substrings/regex against user messages.
# ---------------------------------------------------------------------------
INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore (all )?(the )?(previous|prior|above|earlier) (instructions|prompts?|messages?)",
    r"disregard (all )?(the )?(previous|prior|above|earlier)",
    r"reveal (your |the )?(system )?(prompt|instructions)",
    r"show me (your |the )?(system )?(prompt|instructions)",
    r"what (is|are) your (system )?(prompt|instructions)",
    r"repeat (your |the )?(system )?(prompt|instructions)",
    r"print (your |the )?(system )?(prompt|instructions)",
    r"you are now\b",
    r"you're now\b",
    r"act as (?!an? shl)",  # "act as ..." unless it's about SHL usage
    r"pretend (to be|you are)\b",
    r"forget (all |everything )?(your |the )?(previous|prior|above)",
    r"new instructions?:",
    r"system prompt",
    r"jailbreak",
    r"developer mode",
)


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of all runtime configuration."""

    # --- Data source (BUILD_SPEC §2) ---
    catalog_url: str = _env(
        "CATALOG_URL",
        "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json",
    )
    raw_catalog_path: Path = _path("RAW_CATALOG_PATH", "data/shl_catalog_raw.json")
    normalized_catalog_path: Path = _path(
        "NORMALIZED_CATALOG_PATH", "data/catalog_normalized.json"
    )

    # --- Eval assets (BUILD_SPEC §9) ---
    traces_dir: Path = _path("TRACES_DIR", "eval/traces")
    traces_parsed_path: Path = _path("TRACES_PARSED_PATH", "eval/traces_parsed.json")
    eval_base_url: str = _env("EVAL_BASE_URL", "http://127.0.0.1:8000")
    # Delay between eval /chat calls. On a free LLM tier the eval hammers the
    # provider's tokens-per-minute limit; pacing keeps the dev measurement clean.
    # (This is a dev-harness knob only; it does not affect production latency.)
    eval_turn_delay_s: float = float(_env("EVAL_TURN_DELAY_S", "7"))

    # --- HTTP (catalog download) ---
    http_timeout_s: float = float(_env("HTTP_TIMEOUT_S", "60"))

    # --- Retrieval (BUILD_SPEC §5) ---
    # We run BM25 per user-turn AND on the combined query, then fuse with
    # Reciprocal Rank Fusion (RRF). Per-turn retrieval surfaces each skill from
    # the turn that mentions it (a single concatenated query buries single-skill
    # tests behind broadly-matching items). `pool_size` is how many fused
    # candidates go to the LLM; `per_query_k` is how deep each sub-query reaches.
    retrieval_top_k: int = int(_env("RETRIEVAL_TOP_K", "30"))          # per sub-query depth
    retrieval_per_query_k: int = int(_env("RETRIEVAL_PER_QUERY_K", "25"))
    # 35 preserves the full BM25 recall ceiling (measured) while keeping the
    # prompt small — important for the free-tier tokens-per-minute budget.
    retrieval_pool_size: int = int(_env("RETRIEVAL_POOL_SIZE", "35"))  # candidates to LLM
    rrf_k: int = int(_env("RRF_K", "60"))                             # RRF damping constant
    # Field boosts: the BM25 document repeats name/keys tokens so a query term
    # that matches an item's NAME (a strong signal, e.g. "SQL", "Docker") ranks
    # above items that only mention it in prose.
    name_boost: int = int(_env("NAME_BOOST", "3"))
    keys_boost: int = int(_env("KEYS_BOOST", "2"))
    # Recurring default instruments added to the candidate POOL so the policy can
    # layer them when appropriate. They are a retrieval boost, NOT a forcing
    # function — the LLM decides whether to include them and must honor removals.
    anchor_names: tuple[str, ...] = (
        "Occupational Personality Questionnaire OPQ32r",
        "SHL Verify Interactive G+",
        "Graduate Scenarios",
        "Dependability and Safety Instrument (DSI)",
    )

    # --- LLM provider (BUILD_SPEC §6, Gemini via its OpenAI-compatible endpoint) ---
    # Default is Gemini 2.5 Flash: free tier allows 15 req/min and 1M tokens/min,
    # so it easily survives the evaluator's many-trace replay (Groq's free tier
    # throttled us on both TPM and TPD). Any OpenAI-compatible provider works —
    # point LLM_BASE_URL / LLM_MODEL / the key env var elsewhere to switch.
    llm_base_url: str = _env(
        "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    llm_model: str = _env("LLM_MODEL", "gemini-2.5-flash")
    # The API key is read from the env var named by LLM_API_KEY_ENV (default
    # GEMINI_API_KEY), falling back to GROQ_API_KEY for backwards compatibility.
    llm_api_key: str = _env(_env("LLM_API_KEY_ENV", "GEMINI_API_KEY"), "") or _env(
        "GROQ_API_KEY", ""
    )
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0"))
    llm_timeout_s: float = float(_env("LLM_TIMEOUT_S", "30"))  # Gemini first call can be slower
    llm_max_retries: int = int(_env("LLM_MAX_RETRIES", "2"))
    # Output cap: the JSON reply + up to 10 ids fits comfortably.
    llm_max_tokens: int = int(_env("LLM_MAX_TOKENS", "1200"))
    # Gemini 2.5 Flash is a "thinking" model: reasoning tokens are billed against
    # the output budget and can TRUNCATE the JSON. "none" disables thinking for a
    # clean, complete JSON object (and lower latency). Only sent when non-empty,
    # so switching to a provider that rejects the param is a one-line .env change.
    llm_reasoning_effort: str = _env("LLM_REASONING_EFFORT", "none")
    # On HTTP 429, honor the provider's retry hint up to this cap (kept small so a
    # retry still fits inside the 30s /chat budget); use the default when no hint.
    llm_backoff_cap_s: float = float(_env("LLM_BACKOFF_CAP_S", "12"))
    llm_backoff_default_s: float = float(_env("LLM_BACKOFF_DEFAULT_S", "5"))

    # --- Agent policy (BUILD_SPEC §6.2, §8) ---
    # After this many assistant turns already in the history, stop clarifying and
    # force a shortlist (bounds clarify loops within the 8-turn cap).
    max_clarify_turns: int = int(_env("MAX_CLARIFY_TURNS", "2"))
    max_recommendations: int = int(_env("MAX_RECOMMENDATIONS", "10"))
    # How many top candidates to fall back to if a recommend/refine turn yields
    # no valid ids (BUILD_SPEC §8.3 — never return an empty recommend).
    fallback_recommend_n: int = int(_env("FALLBACK_RECOMMEND_N", "3"))


# Single shared config instance imported across the app.
CONFIG = Config()
