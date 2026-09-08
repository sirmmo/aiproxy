"""Backend registry and factory."""

from __future__ import annotations

from ..config import BackendConfig
from .anthropic_backend import AnthropicBackend
from .base import Backend, Completion, StreamEvent, ToolCall, completion_confidence
from .openai_backend import OpenAIBackend

__all__ = [
    "Backend",
    "Completion",
    "StreamEvent",
    "ToolCall",
    "build_backend",
    "completion_confidence",
]


def build_backend(cfg: BackendConfig) -> Backend:
    if cfg.kind == "anthropic":
        return AnthropicBackend(cfg)
    return OpenAIBackend(cfg)
