"""Thin LLM client (BUILD_SPEC §6, §11 Phase 2).

Talks to Groq's OpenAI-compatible chat-completions endpoint in JSON mode. Kept
deliberately small: one request, a bounded timeout (inside the 30s /chat cap),
and a single retry. All knobs (model, key, temperature, timeout, retries) come
from config/.env — nothing hardcoded.

Callers must handle `LLMError` (agent.py degrades to a safe fallback so we never
500 and never fake success).
"""
from __future__ import annotations

import json
import logging
import re
import time

import requests

from app.config import CONFIG

log = logging.getLogger(__name__)

# Parse a retry hint from a 429 body: Groq's "try again in 12.3s" or Gemini's
# RetryInfo "retryDelay": "30s".
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s")
_RETRY_DELAY_RE = re.compile(r'retryDelay"?\s*:\s*"?([\d.]+)s')


class LLMError(Exception):
    """Raised on any failure to obtain valid JSON from the model."""


def _retry_delay(resp: requests.Response) -> float:
    """Seconds to wait before retrying a 429, capped to stay inside the budget."""
    hdr = resp.headers.get("retry-after")
    delay = 0.0
    if hdr:
        try:
            delay = float(hdr)
        except ValueError:
            delay = 0.0
    if not delay:
        body = resp.text or ""
        m = _RETRY_AFTER_RE.search(body) or _RETRY_DELAY_RE.search(body)
        delay = float(m.group(1)) if m else CONFIG.llm_backoff_default_s
    return min(delay + 0.3, CONFIG.llm_backoff_cap_s)


def chat_json(system_prompt: str, history: list[dict[str, str]]) -> dict:
    """Call the LLM and return the parsed JSON object it emits.

    `history` is the conversation as [{role, content}] with roles user/assistant.
    Raises LLMError on missing key, HTTP error, timeout, or unparseable output.
    """
    if not CONFIG.llm_api_key:
        # Key env var is configurable (GEMINI_API_KEY by default); see config.py.
        raise LLMError("LLM API key is not configured")

    # Gemini's OpenAI-compatible endpoint uses the same Bearer-token auth as Groq,
    # so no header change is needed — only the base URL, model, and key differ.
    url = CONFIG.llm_base_url.rstrip("/") + "/chat/completions"
    messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    payload = {
        "model": CONFIG.llm_model,
        "messages": messages,
        "temperature": CONFIG.llm_temperature,
        "max_tokens": CONFIG.llm_max_tokens,
        # Provider JSON mode: guarantees the content is a single JSON object.
        "response_format": {"type": "json_object"},
    }
    # Disable Gemini 2.5 "thinking" so reasoning tokens don't truncate the JSON.
    # `reasoning_effort` is Gemini-specific, so only send it to the Gemini
    # endpoint — other providers (e.g. Groq) reject the param with a 400.
    if CONFIG.llm_reasoning_effort and "generativelanguage" in CONFIG.llm_base_url:
        payload["reasoning_effort"] = CONFIG.llm_reasoning_effort
    headers = {
        "Authorization": f"Bearer {CONFIG.llm_api_key}",
        "Content-Type": "application/json",
    }

    last_err: Exception | None = None
    # 1 initial attempt + llm_max_retries retries.
    for attempt in range(CONFIG.llm_max_retries + 1):
        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=CONFIG.llm_timeout_s
            )
            if resp.status_code == 429:
                # Rate limited: back off per the provider hint, then retry (if
                # retries remain), staying within the /chat time budget.
                last_err = LLMError(f"HTTP 429: {resp.text[:200]}")
                if attempt < CONFIG.llm_max_retries:
                    delay = _retry_delay(resp)
                    log.warning("LLM 429; backing off %.1fs (attempt %d)", delay, attempt + 1)
                    time.sleep(delay)
                continue
            if resp.status_code != 200:
                last_err = LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                log.warning("LLM attempt %d non-200: %s", attempt + 1, last_err)
                continue
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise LLMError(f"expected JSON object, got {type(parsed).__name__}")
            return parsed
        except Exception as e:  # noqa: BLE001 — retry then surface as LLMError
            last_err = e
            log.warning("LLM attempt %d failed: %r", attempt + 1, e)

    raise LLMError(f"LLM call failed after {CONFIG.llm_max_retries + 1} attempts: {last_err!r}")
