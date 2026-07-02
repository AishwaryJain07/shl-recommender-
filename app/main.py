"""FastAPI service (BUILD_SPEC §7).

Two endpoints:
  * GET  /health -> {"status": "ok"} instantly (no heavy loading, cold-start safe).
  * POST /chat   -> stateless: takes the full history, returns the next reply +
                    a shortlist when appropriate. Wrapped so it never 500s.

The retriever/catalog are lazily built on the first /chat (see agent.get_retriever),
so /health responds immediately even during cold start.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app import agent
from app.schema import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="SHL Conversational Assessment Recommender", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Trivial readiness check — must not trigger model/catalog loading."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run the agent flow for the given conversation history.

    Any internal failure degrades to a valid clarify response (never 500,
    never a fabricated shortlist).
    """
    try:
        return agent.respond(request.messages)
    except Exception as e:  # noqa: BLE001 — belt-and-suspenders over agent's own guard
        log.exception("chat() failed: %r", e)
        return ChatResponse(
            reply=(
                "Sorry, I hit a problem. Could you restate the role and key skills "
                "you're hiring for, and I'll suggest SHL assessments?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )


# Optional: make malformed request bodies (that fail pydantic) return a valid,
# non-500 chat response rather than FastAPI's default 422 (§8: never break schema).
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi import Request  # noqa: E402


@app.exception_handler(RequestValidationError)
async def _on_validation_error(request: Request, exc: RequestValidationError):
    log.warning("Request validation error: %s", exc)
    return JSONResponse(
        status_code=200,
        content=ChatResponse(
            reply=(
                "I couldn't read that request. Please send a messages list of "
                "{role, content} items describing the role you're hiring for."
            ),
            recommendations=[],
            end_of_conversation=False,
        ).model_dump(),
    )
