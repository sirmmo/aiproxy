"""Native Anthropic (Messages API) backend.

Translates the canonical OpenAI-shaped chat messages the agent loop produces into
Anthropic's ``/v1/messages`` request, and normalizes the response (and streaming
events) back into the shared :mod:`base` shapes.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx

from ..config import BackendConfig
from .base import Completion, StreamEvent, ToolCall

_STOP_REASON_MAP = {
    "tool_use": "tool_calls",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "pause_turn": "stop",
}


def _content_to_blocks(content: Any) -> list[dict[str, Any]]:
    """Normalize an OpenAI message ``content`` into Anthropic content blocks."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content:
        ptype = part.get("type")
        if ptype == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                header, _, b64 = url.partition(",")
                media_type = header[5:].split(";")[0] or "image/png"
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    }
                )
            else:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    return blocks


def _translate_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Return (system_prompt, anthropic_messages) with same-role turns merged."""
    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []  # {"role", "content": [blocks]}

    def push(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": list(blocks)})

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            content = msg.get("content")
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content") or "",
            }
            push("user", [block])
            continue
        if role == "assistant":
            blocks = _content_to_blocks(msg.get("content"))
            for tc in msg.get("tool_calls") or []:
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                blocks.append(
                    {"type": "tool_use", "id": tc["id"], "name": fn["name"], "input": args}
                )
            push("assistant", blocks)
            continue
        # user (and any unknown role treated as user)
        push("user", _content_to_blocks(msg.get("content")))

    return "\n\n".join(system_parts), turns


def _translate_tools(tools: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    if not tools:
        return None
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}},
        }
        for t in tools
    ]


def _usage(u: dict[str, Any]) -> dict[str, int]:
    inp = u.get("input_tokens", 0) or 0
    out = u.get("output_tokens", 0) or 0
    return {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out}


class AnthropicBackend:
    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.timeout, connect=15.0),
            base_url=cfg.base_url.rstrip("/"),
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            **self.cfg.default_headers,
        }
        if self.cfg.api_key:
            headers["x-api-key"] = self.cfg.api_key
        return headers

    def _payload(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        params: dict[str, Any],
        stream: bool,
    ) -> dict[str, Any]:
        system, anth_messages = _translate_messages(messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": anth_messages,
            "max_tokens": params.get("max_tokens") or 4096,
        }
        if system:
            payload["system"] = system
        if params.get("temperature") is not None:
            payload["temperature"] = params["temperature"]
        if params.get("top_p") is not None:
            payload["top_p"] = params["top_p"]
        stop = params.get("stop")
        if stop:
            payload["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
        anth_tools = _translate_tools(tools)
        if anth_tools:
            payload["tools"] = anth_tools
            payload["tool_choice"] = {"type": "auto"}
        if stream:
            payload["stream"] = True
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
        resp = await self._client.post("/messages", json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=json.dumps(block.get("input") or {}),
                    )
                )
        return Completion(
            content="".join(text_parts) or None,
            tool_calls=tool_calls,
            finish_reason=_STOP_REASON_MAP.get(data.get("stop_reason"), "stop"),
            usage=_usage(data.get("usage") or {}),
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
            "POST", "/messages", json=payload, headers=self._headers()
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")
                raise httpx.HTTPStatusError(body, request=resp.request, response=resp)
            usage: dict[str, int] = {}
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_raw = line[len("data:") :].strip()
                if not data_raw:
                    continue
                try:
                    event = json.loads(data_raw)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "message_start":
                    usage = _usage((event.get("message") or {}).get("usage") or {})
                elif etype == "content_block_start":
                    block = event.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        yield StreamEvent(
                            type="tool_call",
                            index=event.get("index", 0),
                            id=block.get("id"),
                            name=block.get("name"),
                            arguments="",
                        )
                elif etype == "content_block_delta":
                    delta = event.get("delta") or {}
                    dtype = delta.get("type")
                    if dtype == "text_delta":
                        yield StreamEvent(type="content", text=delta.get("text", ""))
                    elif dtype == "input_json_delta":
                        yield StreamEvent(
                            type="tool_call",
                            index=event.get("index", 0),
                            arguments=delta.get("partial_json", ""),
                        )
                elif etype == "message_delta":
                    d = event.get("delta") or {}
                    out = (event.get("usage") or {}).get("output_tokens")
                    if out is not None:
                        usage["completion_tokens"] = out
                        usage["total_tokens"] = usage.get("prompt_tokens", 0) + out
                    if d.get("stop_reason"):
                        yield StreamEvent(
                            type="finish",
                            finish_reason=_STOP_REASON_MAP.get(d["stop_reason"], "stop"),
                        )
                elif etype == "message_stop":
                    if usage:
                        yield StreamEvent(type="usage", usage=usage)

    async def aclose(self) -> None:
        await self._client.aclose()
