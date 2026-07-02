"""Backend abstraction.

The agent loop speaks one canonical dialect — OpenAI *chat* message dicts for
input, and the normalized shapes below for output. Each backend translates that
canonical form to/from its provider wire format, so the loop never has to know
whether it is driving OpenAI-compatible or native Anthropic upstreams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, as OpenAI represents function arguments


@dataclass
class Completion:
    """Result of a non-streaming turn."""

    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class StreamEvent:
    """One normalized streaming event.

    type == "content": ``text`` holds a content delta.
    type == "tool_call": ``index`` + ``id``/``name``/``arguments`` deltas.
    type == "finish": ``finish_reason`` is set.
    type == "usage": ``usage`` is set.
    """

    type: str
    text: Optional[str] = None
    index: Optional[int] = None
    id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[dict[str, int]] = None


# Whitelisted, provider-neutral sampling params extracted from the client request.
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "response_format",
)


class Backend(Protocol):
    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        params: dict[str, Any],
    ) -> Completion: ...

    def stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        params: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]: ...

    async def aclose(self) -> None: ...
