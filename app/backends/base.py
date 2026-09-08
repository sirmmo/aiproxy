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
    # Non-standard top-level fields the upstream returned (``x_needle``,
    # ``x_mobilemoe``, ...). The agent loop reads a confidence score from here
    # when the completion came from a tool backend.
    extras: dict[str, Any] = field(default_factory=dict)


def completion_confidence(completion: "Completion") -> Optional[float]:
    """Confidence an upstream attached to its tool calls, if it reports one.

    Looks for a ``confidence`` number at the top level of ``extras`` or inside
    any ``x_*`` object (needle-openai puts it in ``x_needle.confidence``).
    Returns ``None`` when the upstream reports nothing, which the loop treats as
    "trusted" so ordinary tool-calling LLMs work unchanged as tool backends.
    """
    candidates: list[Any] = [completion.extras.get("confidence")]
    for key, value in completion.extras.items():
        if key.startswith("x_") and isinstance(value, dict):
            candidates.append(value.get("confidence"))
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


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
