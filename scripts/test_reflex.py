"""Tests for the reflex backends (no / yes / maybe / random). No network.

Covers polarity resolution, the JSON-schema filler, tool-call behaviour,
streaming deltas, config validation, and a full agent-loop round trip through
the real MCP manager (echo server) with the ``yes`` reflex calling a tool.

Run:  python scripts/test_reflex.py
Exits non-zero on failure.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import agent  # noqa: E402
from app.backends import build_backend  # noqa: E402
from app.backends.reflex_backend import ReflexBackend, SchemaFiller  # noqa: E402
from app.config import AppConfig, AssistantConfig, BackendConfig, MCPServerConfig  # noqa: E402
from app.mcp_manager import MCPManager  # noqa: E402

EX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")

USER = [{"role": "user", "content": "Should we ship on Friday?"}]

SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "ratio": {"type": "number"},
        "verdict": {"type": "string", "enum": ["approve", "reject", "defer"]},
        "note": {"type": "string"},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "owner": {"type": ["string", "null"]},
        "when": {"type": "string", "format": "date"},
        "nested": {"$ref": "#/$defs/inner"},
        "choice": {"anyOf": [{"type": "null"}, {"type": "integer"}]},
        "fixed": {"const": "pinned"},
    },
    "required": ["approved", "score", "ratio", "verdict", "note", "reasons", "owner", "when", "nested", "choice", "fixed"],
    "additionalProperties": False,
    "$defs": {
        "inner": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}, "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2}},
            "required": ["ok", "tags"],
            "additionalProperties": False,
        }
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo__add",
            "description": "add",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "echo__echo", "description": "echo", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}},
    },
]


def reflex(kind: str, **kw) -> ReflexBackend:
    backend = build_backend(BackendConfig(kind=kind, **kw))
    assert isinstance(backend, ReflexBackend), type(backend)
    return backend


# --------------------------------------------------------------------------- #
async def test_plain_text():
    for kind, text in (("no", "No."), ("yes", "Yes."), ("maybe", "Maybe.")):
        c = await reflex(kind).complete(None, USER, None, {})
        assert c.content == text and c.finish_reason == "stop" and not c.tool_calls, (kind, c)
        assert c.usage["total_tokens"] == c.usage["prompt_tokens"] + c.usage["completion_tokens"] > 0
        assert c.extras["x_reflex"] == {"kind": kind, "polarity": kind}
    c = await reflex("no", text="Computer says no.").complete(None, USER, None, {})
    assert c.content == "Computer says no."
    print("ok  plain text")


async def test_json_object():
    for kind, answer in (("no", False), ("yes", True), ("maybe", None)):
        c = await reflex(kind).complete(None, USER, None, {"response_format": {"type": "json_object"}})
        doc = json.loads(c.content)
        assert doc["answer"] is answer, (kind, doc)
    print("ok  json_object")


async def test_schema_filler():
    rf = {"type": "json_schema", "json_schema": {"name": "review", "schema": SCHEMA}}
    no = json.loads((await reflex("no").complete(None, USER, None, {"response_format": rf})).content)
    assert no == {
        "approved": False,
        "score": 0,
        "ratio": 0,
        "verdict": "reject",
        "note": "no",
        "reasons": [],
        "owner": "no",
        "when": "1970-01-01",
        "nested": {"ok": False, "tags": ["no", "no"]},
        "choice": 0,
        "fixed": "pinned",
    }, no
    yes = json.loads((await reflex("yes").complete(None, USER, None, {"response_format": rf})).content)
    assert yes["approved"] is True and yes["score"] == 10 and yes["ratio"] == 1, yes
    assert yes["verdict"] == "approve" and yes["reasons"] == ["yes"] and yes["nested"]["ok"] is True, yes
    assert yes["when"] == "9999-12-31" and yes["choice"] == 1, yes
    maybe = json.loads((await reflex("maybe", seed=1).complete(None, USER, None, {"response_format": rf})).content)
    assert maybe["score"] == 5 and maybe["ratio"] == 0.5 and maybe["verdict"] == "defer", maybe
    assert maybe["owner"] is None and maybe["choice"] is None and isinstance(maybe["approved"], bool), maybe
    assert maybe["nested"]["tags"] == ["maybe", "maybe"], maybe
    # Keys present in every document, so strict clients validate.
    for doc in (no, yes, maybe):
        assert set(doc) == set(SCHEMA["properties"]), doc
    # Bare `schema` (some clients) and a missing schema both still yield JSON.
    bare = json.loads((await reflex("no").complete(None, USER, None, {"response_format": {"type": "json_schema", "schema": {"type": "boolean"}}})).content)
    assert bare is False
    fallback = json.loads((await reflex("yes").complete(None, USER, None, {"response_format": {"type": "json_schema"}})).content)
    assert fallback == {"answer": True}
    # Recursive schemas terminate.
    rec = {"type": "object", "properties": {"child": {"$ref": "#"}, "v": {"type": "boolean"}}}
    SchemaFiller(rec, "yes", __import__("random").Random(0)).fill(rec)
    print("ok  json_schema filler")


async def test_tools():
    # no / maybe never call tools.
    for kind in ("no", "maybe"):
        c = await reflex(kind).complete(None, USER, TOOLS, {})
        assert not c.tool_calls and c.content, (kind, c)
    # yes calls the first tool with schema-valid arguments...
    c = await reflex("yes").complete(None, USER, TOOLS, {})
    assert c.finish_reason == "tool_calls" and c.content is None, c
    assert [tc.name for tc in c.tool_calls] == ["echo__add"], c.tool_calls
    assert json.loads(c.tool_calls[0].arguments) == {"a": 1, "b": 1}, c.tool_calls[0].arguments
    # ...or all of them, or none.
    c = await reflex("yes", tool_calls="all").complete(None, USER, TOOLS, {})
    assert [tc.name for tc in c.tool_calls] == ["echo__add", "echo__echo"], c.tool_calls
    assert json.loads(c.tool_calls[1].arguments) == {"text": "yes"}
    c = await reflex("yes", tool_calls="none").complete(None, USER, TOOLS, {})
    assert not c.tool_calls and c.content == "Yes."
    # Once a tool result is in view it answers in words, so the loop terminates.
    after = USER + [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x", "type": "function", "function": {"name": "echo__add", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "x", "name": "echo__add", "content": "2.0"},
    ]
    c = await reflex("yes").complete(None, after, TOOLS, {})
    assert not c.tool_calls and c.content == "Yes.", c
    print("ok  tools")


async def test_random():
    seeded = reflex("random", seed=7)
    first = [seeded.polarity() for _ in range(20)]
    assert set(first) <= {"no", "yes", "maybe"} and len(set(first)) > 1, first
    again = reflex("random", seed=7)
    assert first == [again.polarity() for _ in range(20)], "seed must reproduce"
    only_no = reflex("random", weights={"no": 1, "yes": 0, "maybe": 0})
    assert all(only_no.polarity() == "no" for _ in range(50))
    c = await only_no.complete(None, USER, None, {})
    assert c.content == "No." and c.extras["x_reflex"] == {"kind": "random", "polarity": "no"}
    print("ok  random")


async def test_stream():
    events = [ev async for ev in reflex("yes", text="Yes, absolutely yes.").stream(None, USER, None, {})]
    text = "".join(ev.text for ev in events if ev.type == "content")
    assert text == "Yes, absolutely yes.", text
    assert sum(1 for ev in events if ev.type == "content") == 3, events
    assert [ev.type for ev in events][-2:] == ["finish", "usage"], events
    events = [ev async for ev in reflex("yes").stream(None, USER, TOOLS, {})]
    calls = [ev for ev in events if ev.type == "tool_call"]
    assert calls and calls[0].name == "echo__add" and json.loads(calls[1].arguments) == {"a": 1, "b": 1}, calls
    assert not any(ev.type == "content" for ev in events)
    assert next(ev for ev in events if ev.type == "finish").finish_reason == "tool_calls"
    print("ok  streaming")


def test_config():
    cfg = AppConfig(
        backends={"naas": {"kind": "no"}, "openai": {"kind": "openai", "base_url": "http://x/v1"}},
        assistants=[{"name": "no", "backend": "naas"}],
    )
    assert cfg.backends["naas"].is_reflex and not cfg.backends["openai"].is_reflex
    # YAML 1.1 booleans for the bare words.
    yaml_ish = BackendConfig(kind=True, weights={True: 2, False: 1})
    assert yaml_ish.kind == "yes" and yaml_ish.weights == {"yes": 2, "no": 1}, yaml_ish
    assert BackendConfig(kind=False).kind == "no"
    assert cfg.assistants[0].model is None
    for bad, msg in (
        ({"backends": {"o": {"kind": "openai"}}}, "requires 'base_url'"),
        ({"backends": {"r": {"kind": "random", "weights": {"sometimes": 1}}}}, "weights accepts only"),
        ({"backends": {"o": {"kind": "openai", "base_url": "http://x"}}, "assistants": [{"name": "a", "backend": "o"}]}, "needs 'model'"),
        ({"backends": {"n": {"kind": "no", "tool_calls": "some"}}}, "tool_calls"),
    ):
        try:
            AppConfig(**bad)
        except ValueError as exc:
            assert msg in str(exc), (msg, exc)
        else:
            raise AssertionError(f"config accepted: {bad}")
    print("ok  config validation")


async def test_agent_loop_with_echo_tools():
    """The yes reflex drives the real agent loop: tool call, MCP round trip, answer."""
    manager = MCPManager(
        {"echo": MCPServerConfig(transport="stdio", command=sys.executable, args=[os.path.join(EX_DIR, "echo_mcp_server.py")])}
    )
    try:
        await manager.ensure_started(["echo"])
        toolset = manager.build_toolset(["echo"])
        toolset.restrict(["echo__add"], {}, {})
        assistant = AssistantConfig(name="yes-tools", backend="yaas", mcp_servers=["echo"], max_tool_iterations=2)
        backend = reflex("yes")

        result = await agent.run(assistant, backend, toolset, USER, {})
        assert result["choices"][0]["message"]["content"] == "Yes.", result
        assert result["choices"][0]["finish_reason"] == "stop", result
        assert result["model"] == "yes-tools" and result["usage"]["total_tokens"] > 0, result

        chunks = []
        async for line in agent.run_stream(assistant, backend, toolset, USER, {}):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunks.append(json.loads(line[len("data: "):]))
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        assert text == "Yes.", text
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop", chunks[-1]

        # The no reflex ignores the tools entirely.
        result = await agent.run(AssistantConfig(name="no", backend="naas", mcp_servers=["echo"]), reflex("no"), toolset, USER, {})
        assert result["choices"][0]["message"]["content"] == "No.", result
        print("ok  agent loop with echo MCP tools (yes calls echo__add, then answers)")
    finally:
        await manager.shutdown()


async def main():
    await test_plain_text()
    await test_json_object()
    await test_schema_filler()
    await test_tools()
    await test_random()
    await test_stream()
    test_config()
    await test_agent_loop_with_echo_tools()
    print("\nALL REFLEX TESTS PASSED")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        sys.exit(1)
