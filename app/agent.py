"""The agentic loop: drive a backend LLM through rounds of MCP tool use.

Given an assistant config, a backend and a toolset, this runs the classic loop —
call the model, if it asks for tools execute them against the MCP servers, feed
the results back, repeat — until the model produces a final answer or the
iteration budget is exhausted. It renders both a blocking OpenAI ``chat.completion``
object and an SSE stream of ``chat.completion.chunk`` objects.

An assistant may split the two jobs across models: a ``tool_backend`` decides
which tools to call (gated on the confidence it reports) and ``backend`` writes
the answer once the results are in. See ``_decide``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional

from .backends import Backend, Completion, ToolCall, completion_confidence
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


def _clip(text: str, limit: Optional[int]) -> str:
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated: {len(text) - limit} more characters]"


async def _execute_tool_calls(
    toolset: ToolSet, tool_calls: list[ToolCall], max_chars: Optional[int] = None
) -> list[dict[str, Any]]:
    """Run all requested tool calls concurrently, preserving order in the output."""
    results = await asyncio.gather(*(_run_tool(toolset, tc) for tc in tool_calls))
    return [
        {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": _clip(result, max_chars)}
        for tc, result in zip(tool_calls, results)
    ]



# --------------------------------------------------------------------------- #
# Two-model assistants: a tool backend decides, the answer backend talks
# --------------------------------------------------------------------------- #
def _tool_view(assistant: AssistantConfig, msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The messages the tool backend is shown.

    ``turn``: the system prompt plus everything from the last user message on,
    which includes this turn's tool calls and results. Specialists like needle
    truncate long inputs silently and answer confidently wrong, so they get the
    smallest view that still carries the current turn. ``full``: everything.
    """
    if assistant.tool_context == "full":
        return list(msgs)
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=None)
    if last_user is None:
        return list(msgs)
    head = [m for m in msgs[:last_user] if m.get("role") == "system"]
    return head + msgs[last_user:]


def _rounds_exhausted(assistant: AssistantConfig, round_index: int) -> bool:
    """True once the tool backend has used its ``tool_max_rounds`` for this turn."""
    limit = assistant.tool_max_rounds
    return limit is not None and round_index >= limit


async def _decide(
    assistant: AssistantConfig,
    tool_backend: Backend,
    msgs: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[Completion, Optional[float], bool]:
    """Ask the tool backend for this round's tool calls.

    Returns the completion, the confidence it reported (``None`` if it reports
    none) and whether the calls clear ``tool_confidence`` and should run.
    """
    decision = await tool_backend.complete(
        assistant.tool_model or assistant.model, _tool_view(assistant, msgs), tools, {}
    )
    confidence = completion_confidence(decision)
    execute = bool(decision.tool_calls) and (
        confidence is None or confidence >= assistant.tool_confidence
    )
    logger.info(
        "tool backend decided %s (confidence=%s, execute=%s)",
        [tc.name for tc in decision.tool_calls] or "no call",
        confidence,
        execute,
    )
    return decision, confidence, execute


def _decision_record(
    round_index: int,
    decision: Completion,
    confidence: Optional[float],
    executed: bool,
    fallback: Optional[list[ToolCall]] = None,
) -> dict[str, Any]:
    record = {
        "round": round_index,
        "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in decision.tool_calls],
        "confidence": confidence,
        "executed": executed,
    }
    if fallback:
        record["fallback"] = [{"name": tc.name, "arguments": tc.arguments} for tc in fallback]
    return record


def _last_user_text(msgs: list[dict[str, Any]]) -> str:
    for m in reversed(msgs):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content or "")
    return ""


def _fallback_calls(assistant: AssistantConfig, msgs: list[dict[str, Any]]) -> list[ToolCall]:
    """The forced call for a turn whose first decision ran nothing, or ``[]``."""
    fb = assistant.tool_fallback
    if fb is None:
        return []
    user = _last_user_text(msgs)
    args = {
        k: v.replace("{user}", user) if isinstance(v, str) else v for k, v in fb.arguments.items()
    }
    logger.info("tool fallback %s args=%s", fb.tool, args)
    return [ToolCall(id="call_fallback", name=fb.tool, arguments=json.dumps(args))]


# --------------------------------------------------------------------------- #
# Non-streaming
# --------------------------------------------------------------------------- #
async def run(
    assistant: AssistantConfig,
    backend: Backend,
    toolset: Optional[ToolSet],
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    tool_backend: Optional[Backend] = None,
) -> dict[str, Any]:
    msgs = _prepare_messages(assistant, messages)
    tools = toolset.tools if toolset and not toolset.is_empty else None
    usage: dict[str, int] = {}
    last: Optional[Completion] = None
    decisions: list[dict[str, Any]] = []

    for i in range(assistant.max_tool_iterations + 1):
        # On the final permitted iteration, drop tools to force a natural answer.
        allow_tools = tools if i < assistant.max_tool_iterations else None
        if tool_backend is not None and allow_tools and not _rounds_exhausted(assistant, i):
            decision, confidence, execute = await _decide(assistant, tool_backend, msgs, allow_tools)
            _accumulate_usage(usage, decision.usage)
            calls = decision.tool_calls if execute else []
            fallback = _fallback_calls(assistant, msgs) if not execute and i == 0 else []
            decisions.append(_decision_record(i, decision, confidence, execute, fallback))
            if calls or fallback:
                msgs.append(_assistant_message(None, calls or fallback))
                results = await _execute_tool_calls(
                    toolset, calls or fallback, assistant.tool_result_max_chars
                )
                msgs.extend(results)
                if assistant.expose_tool_results:
                    decisions[-1]["results"] = [r["content"] for r in results]
                continue
            # The specialist declined (or was not confident enough) and there is
            # no fallback: the answer backend replies from whatever the
            # conversation holds, without tools.
            allow_tools = None
        elif tool_backend is not None:
            allow_tools = None  # rounds exhausted: the answer backend never sees tools
        last = await backend.complete(assistant.model, msgs, allow_tools, params)
        _accumulate_usage(usage, last.usage)
        if last.tool_calls and allow_tools:
            msgs.append(_assistant_message(last.content, last.tool_calls))
            msgs.extend(await _execute_tool_calls(toolset, last.tool_calls, assistant.tool_result_max_chars))
            continue
        break

    content = last.content if last else ""
    finish = last.finish_reason if last else "stop"
    if finish == "tool_calls":  # budget exhausted mid-tool-use
        finish = "stop"
    result: dict[str, Any] = {
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
    if tool_backend is not None:
        result["x_aiproxy"] = {"tool_backend": assistant.tool_backend, "decisions": decisions}
    return result


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
    tool_backend: Optional[Backend] = None,
) -> AsyncIterator[str]:
    stream_id = _new_id()
    msgs = _prepare_messages(assistant, messages)
    tools = toolset.tools if toolset and not toolset.is_empty else None
    decisions: list[dict[str, Any]] = []

    yield _chunk(stream_id, assistant.name, {"role": "assistant", "content": ""})
    try:
        for i in range(assistant.max_tool_iterations + 1):
            allow_tools = tools if i < assistant.max_tool_iterations else None
            if tool_backend is not None and allow_tools and not _rounds_exhausted(assistant, i):
                # The decision is a short non-streamed call; only the answer streams.
                decision, confidence, execute = await _decide(
                    assistant, tool_backend, msgs, allow_tools
                )
                calls = decision.tool_calls if execute else []
                fallback = _fallback_calls(assistant, msgs) if not execute and i == 0 else []
                decisions.append(_decision_record(i, decision, confidence, execute, fallback))
                if calls or fallback:
                    msgs.append(_assistant_message(None, calls or fallback))
                    results = await _execute_tool_calls(
                        toolset, calls or fallback, assistant.tool_result_max_chars
                    )
                    msgs.extend(results)
                    if assistant.expose_tool_results:
                        decisions[-1]["results"] = [r["content"] for r in results]
                    continue
                allow_tools = None
            elif tool_backend is not None:
                allow_tools = None  # rounds exhausted: the answer backend never sees tools

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
                msgs.extend(await _execute_tool_calls(toolset, tool_calls, assistant.tool_result_max_chars))
                continue

            final = _chunk(stream_id, assistant.name, {}, finish=finish_reason or "stop")
            if tool_backend is not None:
                obj = json.loads(final[len("data: ") :])
                obj["x_aiproxy"] = {"tool_backend": assistant.tool_backend, "decisions": decisions}
                final = f"data: {json.dumps(obj)}\n\n"
            yield final
            yield "data: [DONE]\n\n"
            return
    except Exception as exc:  # noqa: BLE001 - stream an error then close cleanly
        logger.exception("stream failed")
        yield _chunk(stream_id, assistant.name, {"content": f"\n\n[gateway error] {exc}"})
        yield _chunk(stream_id, assistant.name, {}, finish="stop")
        yield "data: [DONE]\n\n"
