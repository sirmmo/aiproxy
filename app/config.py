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


class BackendConfig(BaseModel):
    """An upstream LLM provider."""

    kind: Literal["openai", "anthropic"] = "openai"
    base_url: str
    api_key: Optional[str] = None
    # Extra headers merged into every request (e.g. org id, provider version).
    default_headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = 300.0


class ToolFallbackConfig(BaseModel):
    """A tool call the gateway makes itself when the tool backend's first
    decision of a turn runs nothing (it declined, or was not confident enough).

    ``arguments`` values may contain ``{user}``, replaced by the text of the
    last user message. Pinned ``tool_arguments`` are merged in as for any call.
    """

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AssistantConfig(BaseModel):
    """A virtual model exposed to clients via the OpenAI ``model`` field."""

    name: str
    backend: str
    model: str
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


def load_config(path: str | None = None) -> AppConfig:
    path = path or os.environ.get("CONFIG_PATH", "config.yaml")
    if not os.path.exists(path):
        # An empty config is valid; assistants/servers can be added via the admin API.
        return AppConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _expand_env(raw)
    return AppConfig(**raw)
