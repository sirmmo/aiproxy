"""OpenAI-compatible backend.

Works with any upstream that implements ``POST {base_url}/chat/completions`` in
the OpenAI dialect: OpenAI, Anthropic's compat endpoint, Groq, Together, Mistral,
vLLM, Ollama (``/v1``), LM Studio, OpenRouter, etc.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx

from ..config import BackendConfig
from .base import Completion, StreamEvent, ToolCall

_SAMPLING_PASSTHROUGH = (
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "response_format",
)


class OpenAIBackend:
    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.timeout, connect=15.0),
            base_url=cfg.base_url.rstrip("/"),
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.cfg.default_headers}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def _payload(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        params: dict[str, Any],
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        for key in _SAMPLING_PASSTHROUGH:
            if params.get(key) is not None:
                payload[key] = params[key]
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        extra = params.get("extra_body")
        if extra:
            payload.update(extra)
        return payload

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        params: dict[str, Any],
    ) -> Completion:
        payload = self._payload(model, messages, tools, params, stream=False)
        resp = await self._client.post("/chat/completions", json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        tool_calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments") or "{}",
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        ]
        return Completion(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=data["choices"][0].get("finish_reason") or "stop",
            usage=data.get("usage") or {},
        )

    async def stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        params: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        payload = self._payload(model, messages, tools, params, stream=True)
        async with self._client.stream(
            "POST", "/chat/completions", json=payload, headers=self._headers()
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")
                raise httpx.HTTPStatusError(body, request=resp.request, response=resp)
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    yield StreamEvent(type="usage", usage=chunk["usage"])
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        yield StreamEvent(type="content", text=delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        yield StreamEvent(
                            type="tool_call",
                            index=tc.get("index", 0),
                            id=tc.get("id"),
                            name=fn.get("name"),
                            arguments=fn.get("arguments"),
                        )
                    if choice.get("finish_reason"):
                        yield StreamEvent(type="finish", finish_reason=choice["finish_reason"])

    async def aclose(self) -> None:
        await self._client.aclose()
