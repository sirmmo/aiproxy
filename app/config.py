"""Configuration models and loading.

The gateway is configured by three collections:

* ``mcp_servers``  – reusable MCP server definitions (stdio / sse / http).
* ``backends``     – upstream LLM providers (OpenAI-compatible or native Anthropic).
* ``assistants``   – the *virtual models* clients select via the ``model`` field.
                     Each binds a backend + a system prompt + a set of MCP servers.

Config is read from a YAML file at startup (``CONFIG_PATH``, default ``config.yaml``)
and can be mutated at runtime through the admin API. ``${VAR}`` / ``${VAR:-default}``
placeholders are expanded from the process environment when the file is loaded.
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings."""
    if isinstance(value, str):

        def repl(m: re.Match) -> str:
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else "")

        return _ENV_RE.sub(repl, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


class MCPServerConfig(BaseModel):
    """A reusable MCP server. ``stdio`` spawns a subprocess; ``sse``/``http`` connect to a URL."""

    transport: Literal["stdio", "sse", "http", "streamable-http"] = "stdio"
    # stdio
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = None
    # sse / http
    url: Optional[str] = None
    headers: dict[str, str] = Field(default_factory=dict)
    description: Optional[str] = None


REFLEX_KINDS = ("no", "yes", "maybe", "random")


class BackendConfig(BaseModel):
    """An upstream LLM provider, or a *reflex* that answers without one.

    ``kind`` ``openai`` / ``anthropic`` talk to a provider at ``base_url``.
    ``no`` / ``yes`` / ``maybe`` / ``random`` are reflex backends ("system one"):
    every request is answered with that polarity, shaped to the request (plain
    text, ``response_format`` documents, tool calls, streaming). See
    ``app/backends/reflex_backend.py``.
    """

    kind: Literal["openai", "anthropic", "no", "yes", "maybe", "random"] = "openai"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    # Extra headers merged into every request (e.g. org id, provider version).
    default_headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = 300.0

    # --- reflex backends only ---
    # Override the answer text ("No." / "Yes." / "Maybe." by default).
    text: Optional[str] = None
    # ``random``: relative weights of the three polarities (default: equal).
    weights: dict[str, float] = Field(default_factory=dict)
    # Seed the random generator for reproducible ``random`` / ``maybe`` answers.
    seed: Optional[int] = None
    # ``yes`` with tools on offer: call the ``first`` tool, ``all`` of them, or ``none``.
    tool_calls: Literal["first", "all", "none"] = "first"
    # Simulated think time (seconds) before answering, and pause between streamed words.
    latency: float = Field(default=0.0, ge=0.0)
    stream_delay: float = Field(default=0.0, ge=0.0)

    @property
    def is_reflex(self) -> bool:
        return self.kind in REFLEX_KINDS

    # YAML 1.1 reads a bare `yes` / `no` as a boolean, so `kind: no` arrives as
    # False and `weights: {yes: 1}` as {True: 1}. Accept both spellings.
    @field_validator("kind", mode="before")
    @classmethod
    def _bool_kind(cls, v: Any) -> Any:
        if isinstance(v, bool):
            return "yes" if v else "no"
        return v

    @field_validator("weights", mode="before")
    @classmethod
    def _bool_weight_keys(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {("yes" if k is True else "no" if k is False else k): w for k, w in v.items()}
        return v

    @model_validator(mode="after")
    def _check_fields(self) -> "BackendConfig":
        if not self.is_reflex and not self.base_url:
            raise ValueError(f"backend kind '{self.kind}' requires 'base_url'")
        unknown = set(self.weights) - set(REFLEX_KINDS[:3])
        if unknown:
            raise ValueError(f"weights accepts only no/yes/maybe, got {sorted(unknown)}")
        return self


class ToolFallbackConfig(BaseModel):
    """A tool call the gateway makes itself when the tool backend's first
    decision of a turn runs nothing (it declined, or was not confident enough).

    ``arguments`` values may contain ``{user}``, replaced by the text of the
    last user message. Pinned ``tool_arguments`` are merged in as for any call.
    """

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCitationsConfig(BaseModel):
    """Append the source ids found in this turn's tool results to the answer.

    ``pattern`` is a regex matched against every tool result; distinct matches
    (in order of appearance, at most ``max``) are appended as
    ``\\n\\n<label>: a, b, c`` and reported in ``x_aiproxy.citations``. Small
    answer models do not cite reliably when asked to; the gateway can.
    """

    pattern: str
    label: str = "Sources"
    max: int = Field(default=8, ge=1)
    # Do not append sources when the answer matches this (a refusal such as
    # "the knowledge graph has nothing on that" must not carry citations).
    skip_pattern: Optional[str] = None


class ToolEmptyConfig(BaseModel):
    """Short-circuit when a tool round retrieved nothing.

    When every executed tool result of a round matches ``pattern`` (an empty
    string, ``[]``, ``"passages": []`` ...), the answer model is not called and
    ``reply`` is returned as the assistant message, with
    ``x_aiproxy.empty_retrieval: true``. Deterministic negative rejection for
    retrieval assistants whose answer model tends to fill the gap.
    """

    pattern: str = r"^\s*(\[\s*\]|\{\s*\})?\s*$"
    reply: str = "I found nothing about that in the knowledge base."


class AssistantConfig(BaseModel):
    """A virtual model exposed to clients via the OpenAI ``model`` field."""

    name: str
    backend: str
    # Upstream model id. Optional for reflex backends, which have no model.
    model: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    mcp_servers: list[str] = Field(default_factory=list)
    max_tool_iterations: int = 8
    # Two-model assistants. When ``tool_backend`` is set, every round first asks
    # that backend (a tool-calling specialist such as needle-openai) whether to
    # call a tool. Its calls are executed only when the confidence it reports
    # (``x_needle.confidence`` or a top-level ``confidence``) is at least
    # ``tool_confidence``; a backend that reports no confidence is trusted.
    # Otherwise ``backend`` answers without tools. ``tool_model`` defaults to
    # ``model``. ``tool_context`` is what the tool backend sees: ``turn`` (system
    # prompt + everything from the last user message on, the safe choice for
    # specialists that truncate long inputs silently) or ``full``.
    tool_backend: Optional[str] = None
    tool_model: Optional[str] = None
    tool_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    tool_context: Literal["turn", "full"] = "turn"
    # How many tool rounds the tool backend may run per turn before the answer
    # backend replies. ``None`` allows ``max_tool_iterations``. Specialists that
    # cannot read long inputs (needle) should decide once: after the first
    # round they would be shown the tool results and stall or misfire.
    tool_max_rounds: Optional[int] = Field(default=None, ge=0)
    # Retrieval-style assistants want context on every turn; a specialist that
    # declines (needle scores retrieval calls low) would otherwise leave the
    # answer model to invent one. Only used with ``tool_backend``.
    tool_fallback: Optional[ToolFallbackConfig] = None
    # Include each executed tool's (clipped) result text in x_aiproxy.decisions,
    # for evaluation harnesses that need the retrieved context alongside the
    # answer. Off by default: results can be thousands of characters.
    expose_tool_results: bool = False
    tool_citations: Optional[ToolCitationsConfig] = None
    tool_empty: Optional[ToolEmptyConfig] = None
    # Restrict and re-describe the tools this assistant exposes, by exposed name
    # (``<server>__<tool>``). Small tool-calling models degrade with large tool
    # sets and long descriptions; an allow-list of a few tools with one-line
    # descriptions written for the deciding model is the fix. ``None`` exposes
    # every tool of the attached servers.
    tool_allowlist: Optional[list[str]] = None
    tool_descriptions: dict[str, str] = Field(default_factory=dict)
    # Truncate each tool result to this many characters before it enters the
    # conversation. Retrieval tools can return thousands of tokens; a small
    # answer model with an 8k window (and slow CPU prefill) needs a cap.
    tool_result_max_chars: Optional[int] = Field(default=None, ge=1)
    # Arguments pinned per exposed tool name: hidden from the schema the model
    # sees and merged into every call (the model's value, if any, is replaced).
    # For parameters a small tool-calling model should not choose (page sizes,
    # dataset ids, expansion depth).
    tool_arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Default sampling params; client-supplied values override these per request.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    # Provider-specific body fields merged verbatim into the upstream request.
    extra_body: dict[str, Any] = Field(default_factory=dict)


class ApimanConfig(BaseModel):
    """Validate caller API keys against an Apiman deployment, in parallel to
    the static ``proxy_api_keys`` list.

    Two topologies are supported (pick one with ``mode``):

    * ``gateway_probe`` – aiproxy receives the caller's key and validates it by
      making a cheap request *through* the Apiman gateway to a registered probe
      API. A 2xx means the key is valid for that API + plan; 401/403 means it is
      not. Results are cached for ``cache_ttl`` seconds.
    * ``trusted_header`` – the Apiman gateway is deployed in front of aiproxy and
      injects a shared-secret header (via an "Add Header" policy); aiproxy
      authorizes any request carrying the matching secret and never sees raw keys.
    """

    enabled: bool = False
    mode: Literal["gateway_probe", "trusted_header"] = "gateway_probe"

    # --- gateway_probe ---
    # Base gateway URL, e.g. http://apiman-gw:8080/apiman-gateway
    gateway_url: Optional[str] = None
    # The probe managed API as "{org}/{api}/{version}", e.g. "aiproxy/authcheck/1.0".
    probe_api: Optional[str] = None
    # Path appended after {version}; its backend should return 2xx (e.g. aiproxy /health).
    probe_path: str = ""
    # Header the gateway expects the API key in (Apiman default is X-API-Key).
    api_key_header: str = "X-API-Key"
    cache_ttl: float = 60.0
    timeout: float = 5.0

    # --- trusted_header ---
    header: Optional[str] = None  # e.g. X-Apiman-Gateway-Token
    secret: Optional[str] = None

    @model_validator(mode="after")
    def _check_mode_fields(self) -> "ApimanConfig":
        if not self.enabled:
            return self
        if self.mode == "gateway_probe":
            if not self.gateway_url or not self.probe_api:
                raise ValueError(
                    "apiman.gateway_probe requires 'gateway_url' and 'probe_api'"
                )
            if self.probe_api.count("/") != 2:
                raise ValueError("apiman.probe_api must be '{org}/{api}/{version}'")
        elif self.mode == "trusted_header":
            if not self.header or not self.secret:
                raise ValueError(
                    "apiman.trusted_header requires 'header' and 'secret'"
                )
        return self


class AppConfig(BaseModel):
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    backends: dict[str, BackendConfig] = Field(default_factory=dict)
    assistants: list[AssistantConfig] = Field(default_factory=list)
    # If non-empty, callers of /v1/* must present a matching Bearer token.
    proxy_api_keys: list[str] = Field(default_factory=list)
    # Optional Apiman-backed key validation, checked in parallel to proxy_api_keys.
    apiman: ApimanConfig = Field(default_factory=ApimanConfig)

    @field_validator("proxy_api_keys", mode="before")
    @classmethod
    def _drop_empty_keys(cls, v: Any) -> Any:
        # ${PROXY_API_KEY:-} expands to "" when unset; an empty key must not
        # accidentally enable (and then reject on) auth.
        if isinstance(v, list):
            return [k for k in v if isinstance(k, str) and k.strip()]
        return v

    @model_validator(mode="after")
    def _check_assistant_models(self) -> "AppConfig":
        # A provider backend needs a model id; a reflex backend does not.
        for a in self.assistants:
            backend = self.backends.get(a.backend)
            if backend is not None and not backend.is_reflex and not a.model:
                raise ValueError(
                    f"assistant '{a.name}' uses backend '{a.backend}' ({backend.kind}) and needs 'model'"
                )
        return self


def load_config(path: str | None = None) -> AppConfig:
    path = path or os.environ.get("CONFIG_PATH", "config.yaml")
    if not os.path.exists(path):
        # An empty config is valid; assistants/servers can be added via the admin API.
        return AppConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _expand_env(raw)
    return AppConfig(**raw)
