"""Pydantic request/response models (BUILD_SPEC §7).

The response schema is NON-NEGOTIABLE: exactly {reply, recommendations,
end_of_conversation}, where each recommendation is exactly {name, url,
test_type}. `recommendations` is ALWAYS a list ([] when clarifying/refusing,
never null). `extra="forbid"` on outputs guarantees we never leak extra keys.
Inputs are lenient (`extra="ignore"`) so a slightly-off request never 500s.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """One conversation message from the request history."""

    model_config = ConfigDict(extra="ignore")
    role: str = ""
    content: str = ""


class ChatRequest(BaseModel):
    """POST /chat body — the full, stateless conversation history."""

    model_config = ConfigDict(extra="ignore")
    messages: list[Message] = Field(default_factory=list)


class Recommendation(BaseModel):
    """A single shortlisted assessment. Exactly three fields."""

    model_config = ConfigDict(extra="forbid")
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    """POST /chat response. `recommendations` is always a list (never null)."""

    model_config = ConfigDict(extra="forbid")
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    """GET /health response."""

    model_config = ConfigDict(extra="forbid")
    status: str = "ok"
