"""The agentic loop: drive a backend LLM through rounds of MCP tool use.

Given an assistant config, a backend and a toolset, this runs the classic loop —
call the model, if it asks for tools execute them against the MCP servers, feed
the results back, repeat — until the model produces a final answer or the
iteration budget is exhausted. It renders both a blocking OpenAI ``chat.completion``
object and an SSE stream of ``chat.completion.chunk`` objects.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional

from .backends import Backend, Completion, ToolCall
from .config import AssistantConfig
from .mcp_manager import ToolSet

logger = logging.getLogger("aiproxy.agent")


def _new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _prepare_messages(assistant: AssistantConfig, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    msgs = list(messages)
    if assistant.system_prompt and not (msgs and msgs[0].get("role") == "system"):
        msgs.insert(0, {"role": "system", "content": assistant.system_prompt})
    return msgs


def _assistant_message(content: Optional[str], tool_calls: list[ToolCall]) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in tool_calls
        ]
    return msg


def _accumulate_usage(total: dict[str, int], add: dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if add.get(key):
            total[key] = total.get(key, 0) + int(add[key])


async def _run_tool(toolset: ToolSet, tc: ToolCall) -> str:
    try:
        args = json.loads(tc.arguments or "{}")
        if not isinstance(args, dict):
            args = {"value": args}
    except json.JSONDecodeError:
        return f"[tool error] invalid JSON arguments for '{tc.name}': {tc.arguments!r}"
    try:
        logger.info("tool call %s args=%s", tc.name, args)
        return await toolset.call(tc.name, args)
    except Exception as exc:  # noqa: BLE001 - reported back to the model
        logger.exception("tool '%s' failed", tc.name)
        return f"[tool error] {tc.name}: {exc}"


async def _execute_tool_calls(toolset: ToolSet, tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    """Run all requested tool calls concurrently, preserving order in the output."""
    results = await asyncio.gather(*(_run_tool(toolset, tc) for tc in tool_calls))
    return [
        {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result}
        for tc, result in zip(tool_calls, results)
    ]


# --------------------------------------------------------------------------- #
# Non-streaming
# --------------------------------------------------------------------------- #
async def run(
    assistant: AssistantConfig,
    backend: Backend,
    toolset: Optional[ToolSet],
    messages: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    msgs = _prepare_messages(assistant, messages)
    tools = toolset.tools if toolset and not toolset.is_empty else None
    usage: dict[str, int] = {}
    last: Optional[Completion] = None

    for i in range(assistant.max_tool_iterations + 1):
        # On the final permitted iteration, drop tools to force a natural answer.
        allow_tools = tools if i < assistant.max_tool_iterations else None
        last = await backend.complete(assistant.model, msgs, allow_tools, params)
        _accumulate_usage(usage, last.usage)
        if last.tool_calls and allow_tools:
            msgs.append(_assistant_message(last.content, last.tool_calls))
            msgs.extend(await _execute_tool_calls(toolset, last.tool_calls))
            continue
        break

    content = last.content if last else ""
    finish = last.finish_reason if last else "stop"
    if finish == "tool_calls":  # budget exhausted mid-tool-use
        finish = "stop"
    return {
        "id": _new_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": assistant.name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content or ""},
                "finish_reason": finish,
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
def _chunk(stream_id: str, model: str, delta: dict[str, Any], finish: Optional[str] = None) -> str:
    obj = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(obj)}\n\n"


def _merge_tool_delta(acc: dict[int, dict[str, Any]], ev) -> None:
    slot = acc.setdefault(ev.index or 0, {"id": None, "name": None, "arguments": ""})
    if ev.id:
        slot["id"] = ev.id
    if ev.name:
        slot["name"] = ev.name
    if ev.arguments:
        slot["arguments"] += ev.arguments


def _finalize_tool_calls(acc: dict[int, dict[str, Any]]) -> list[ToolCall]:
    calls = []
    for idx in sorted(acc):
        slot = acc[idx]
        if not slot.get("name"):
            continue
        calls.append(
            ToolCall(
                id=slot["id"] or f"call_{idx}",
                name=slot["name"],
                arguments=slot["arguments"] or "{}",
            )
        )
    return calls


async def run_stream(
    assistant: AssistantConfig,
    backend: Backend,
    toolset: Optional[ToolSet],
    messages: list[dict[str, Any]],
    params: dict[str, Any],
) -> AsyncIterator[str]:
    stream_id = _new_id()
    msgs = _prepare_messages(assistant, messages)
    tools = toolset.tools if toolset and not toolset.is_empty else None

    yield _chunk(stream_id, assistant.name, {"role": "assistant", "content": ""})
    try:
        for i in range(assistant.max_tool_iterations + 1):
            allow_tools = tools if i < assistant.max_tool_iterations else None
            content_parts: list[str] = []
            tool_acc: dict[int, dict[str, Any]] = {}
            finish_reason: Optional[str] = None

            async for ev in backend.stream(assistant.model, msgs, allow_tools, params):
                if ev.type == "content" and ev.text:
                    content_parts.append(ev.text)
                    yield _chunk(stream_id, assistant.name, {"content": ev.text})
                elif ev.type == "tool_call":
                    _merge_tool_delta(tool_acc, ev)
                elif ev.type == "finish":
                    finish_reason = ev.finish_reason

            tool_calls = _finalize_tool_calls(tool_acc)
            if tool_calls and allow_tools:
                msgs.append(_assistant_message("".join(content_parts) or None, tool_calls))
                msgs.extend(await _execute_tool_calls(toolset, tool_calls))
                continue

            yield _chunk(
                stream_id, assistant.name, {}, finish=finish_reason or "stop"
            )
            yield "data: [DONE]\n\n"
            return
    except Exception as exc:  # noqa: BLE001 - stream an error then close cleanly
        logger.exception("stream failed")
        yield _chunk(stream_id, assistant.name, {"content": f"\n\n[gateway error] {exc}"})
        yield _chunk(stream_id, assistant.name, {}, finish="stop")
        yield "data: [DONE]\n\n"
