"""End-to-end smoke test with NO real LLM required.

Exercises the real MCP manager (spawning the stdio echo server), the ToolSet
namespacing/routing, and the full agent loop (both non-streaming and streaming)
against a scripted fake backend that asks for a tool then answers.

Run:  python scripts/smoke_test.py
Exits non-zero on failure.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import agent  # noqa: E402
from app.backends.base import Completion, StreamEvent, ToolCall  # noqa: E402
from app.config import AssistantConfig, MCPServerConfig  # noqa: E402
from app.mcp_manager import MCPManager  # noqa: E402

EX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


class FakeBackend:
    """Turn 1: ask to call the add tool. Turn 2: report the answer."""

    def __init__(self):
        self.calls = 0

    async def complete(self, model, messages, tools, params):
        self.calls += 1
        if self.calls == 1:
            assert tools, "expected tools to be offered on the first turn"
            names = [t["function"]["name"] for t in tools]
            assert "echo__add" in names, f"missing namespaced tool, got {names}"
            return Completion(
                content=None,
                tool_calls=[ToolCall(id="c1", name="echo__add", arguments='{"a": 2, "b": 3}')],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        # Second turn: the tool result must be present in the conversation.
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert tool_msgs and tool_msgs[-1]["content"].strip() == "5.0", (
            f"unexpected tool result: {tool_msgs}"
        )
        return Completion(
            content="The sum is 5.",
            tool_calls=[],
            finish_reason="stop",
            usage={"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
        )

    async def stream(self, model, messages, tools, params):
        self.calls += 1
        if self.calls == 1:
            yield StreamEvent(type="tool_call", index=0, id="c1", name="echo__add", arguments="")
            yield StreamEvent(type="tool_call", index=0, arguments='{"a": 2,')
            yield StreamEvent(type="tool_call", index=0, arguments=' "b": 3}')
            yield StreamEvent(type="finish", finish_reason="tool_calls")
            return
        for piece in ["The ", "sum ", "is ", "5."]:
            yield StreamEvent(type="content", text=piece)
        yield StreamEvent(type="finish", finish_reason="stop")

    async def aclose(self):
        pass


def build_manager() -> MCPManager:
    return MCPManager(
        {
            "echo": MCPServerConfig(
                transport="stdio",
                command=sys.executable,
                args=[os.path.join(EX_DIR, "echo_mcp_server.py")],
            )
        }
    )


ASSISTANT = AssistantConfig(
    name="test-assistant",
    backend="fake",
    model="fake-model",
    system_prompt="You are a calculator.",
    mcp_servers=["echo"],
    max_tool_iterations=4,
)


# --- two-model assistant: a specialist decides, a chat model talks -----------
class FakeDecider:
    """Stands in for needle-openai: emits a tool call with a confidence score,
    never prose. ``confidences`` are handed out one per call, in order."""

    def __init__(self, confidences):
        self.confidences = list(confidences)
        self.seen: list[list[dict]] = []

    async def complete(self, model, messages, tools, params):
        assert tools, "the decider must always be offered tools"
        assert params == {}, f"decider must not receive answer-model params: {params}"
        self.seen.append(messages)
        confidence = self.confidences.pop(0)
        if confidence is None:  # declines: returns its reasoning trace, like needle does
            return Completion(content="No tool applies.", extras={"x_needle": {"confidence": 0.0}})
        return Completion(
            content=None,
            tool_calls=[ToolCall(id="d1", name="echo__add", arguments='{"a": 2, "b": 3}')],
            finish_reason="tool_calls",
            usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            extras={"x_needle": {"confidence": confidence, "type": "call"}},
        )

    async def stream(self, model, messages, tools, params):  # never used for decisions
        raise AssertionError("the tool backend is never streamed")

    async def aclose(self):
        pass


class FakeTalker:
    """Stands in for a chat model that cannot call tools: it must never be offered
    any, and answers from the tool results present in the conversation."""

    def __init__(self):
        self.calls = 0

    def _answer(self, messages, tools):
        assert tools is None, "the answer backend must not be offered tools"
        self.calls += 1
        results = [m["content"].strip() for m in messages if m.get("role") == "tool"]
        return f"Result: {results[-1]}" if results else "No tool was used."

    async def complete(self, model, messages, tools, params):
        return Completion(content=self._answer(messages, tools), usage={"total_tokens": 7})

    async def stream(self, model, messages, tools, params):
        for piece in self._answer(messages, tools).split(" "):
            yield StreamEvent(type="content", text=piece + " ")
        yield StreamEvent(type="finish", finish_reason="stop")

    async def aclose(self):
        pass


TWO_MODEL = AssistantConfig(
    name="two-model",
    backend="talker",
    model="talker-model",
    tool_backend="decider",
    tool_model="needle-2",
    tool_confidence=0.6,
    system_prompt="Answer from tool results.",
    mcp_servers=["echo"],
    max_tool_iterations=3,
)


async def main() -> None:
    print("→ starting MCP echo server...")
    mgr = build_manager()
    await mgr.ensure_started(["echo"])
    toolset = mgr.build_toolset(["echo"])
    tool_names = sorted(t["function"]["name"] for t in toolset.tools)
    assert tool_names == ["echo__add", "echo__echo", "echo__uppercase"], tool_names
    print(f"  tools discovered: {tool_names}")

    print("→ calling MCP tool directly...")
    direct = await toolset.call("echo__uppercase", {"text": "hello"})
    assert direct == "HELLO", repr(direct)
    print(f"  echo__uppercase('hello') = {direct!r}")

    print("→ non-streaming agent loop...")
    resp = await agent.run(
        ASSISTANT, FakeBackend(), toolset, [{"role": "user", "content": "add 2 and 3"}], {}
    )
    content = resp["choices"][0]["message"]["content"]
    assert content == "The sum is 5.", repr(resp)
    assert resp["usage"]["total_tokens"] == 39, resp["usage"]
    assert resp["model"] == "test-assistant"
    print(f"  final: {content!r}  usage={resp['usage']}")

    print("→ streaming agent loop...")
    chunks = []
    async for line in agent.run_stream(
        ASSISTANT, FakeBackend(), toolset, [{"role": "user", "content": "add 2 and 3"}], {}
    ):
        chunks.append(line)
    text = "".join(chunks)
    assert "The " in text and "5." in text, text
    assert text.rstrip().endswith("data: [DONE]"), text[-80:]
    # Reconstruct streamed content.
    streamed = ""
    for line in text.splitlines():
        if line.startswith("data:") and "[DONE]" not in line:
            delta = json.loads(line[5:].strip())["choices"][0]["delta"]
            streamed += delta.get("content", "")
    assert streamed == "The sum is 5.", repr(streamed)
    print(f"  streamed content: {streamed!r}")

    print("→ tool allow-list + description override...")
    narrowed = mgr.build_toolset(["echo"])
    narrowed.restrict(["echo__add", "echo__nope"], {"echo__add": "Add two numbers."})
    assert [t["function"]["name"] for t in narrowed.tools] == ["echo__add"], narrowed.tools
    assert narrowed.tools[0]["function"]["description"] == "Add two numbers."
    assert await narrowed.call("echo__uppercase", {"text": "x"}) == "X"  # routing untouched
    print("  exposes echo__add only, hidden tools still route")
    pinned = mgr.build_toolset(["echo"])
    pinned.restrict(["echo__add"], None, {"echo__add": {"b": 40}})
    assert "b" not in pinned.tools[0]["function"]["parameters"]["properties"]
    assert "b" not in pinned.tools[0]["function"]["parameters"].get("required", [])
    assert (await pinned.call("echo__add", {"a": 2, "b": 3})).strip() == "42.0"  # pinned wins
    print("  pinned argument hidden from the schema and forced on the call")

    print("→ two-model assistant: confident decision runs the tool, talker answers...")
    decider, talker = FakeDecider([0.9, None]), FakeTalker()
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "add 2 and 3"},
    ]
    resp = await agent.run(TWO_MODEL, talker, toolset, history, {"temperature": 0}, tool_backend=decider)
    content = resp["choices"][0]["message"]["content"]
    assert content == "Result: 5.0", repr(resp)
    decisions = resp["x_aiproxy"]["decisions"]
    assert [d["executed"] for d in decisions] == [True, False], decisions
    assert decisions[0]["confidence"] == 0.9 and decisions[0]["tool_calls"][0]["name"] == "echo__add"
    # `turn` context: the decider saw the system prompt and the current turn only.
    first_view = decider.seen[0]
    assert [m["role"] for m in first_view] == ["system", "user"], first_view
    assert "earlier" not in json.dumps(first_view)
    # ...and on the second round it also saw this turn's tool call + result.
    assert [m["role"] for m in decider.seen[1]] == ["system", "user", "assistant", "tool"]
    assert talker.calls == 1
    print(f"  final: {content!r}  decisions={[(d['confidence'], d['executed']) for d in decisions]}")

    print("→ tool results are clipped to tool_result_max_chars...")
    clipped = TWO_MODEL.model_copy(update={"tool_result_max_chars": 2})
    decider, talker = FakeDecider([0.9, None]), FakeTalker()
    resp = await agent.run(clipped, talker, toolset, [{"role": "user", "content": "add 2 and 3"}], {}, tool_backend=decider)
    assert resp["choices"][0]["message"]["content"].startswith("Result: 5.\n[truncated: "), resp
    print("  clipped:", repr(resp["choices"][0]["message"]["content"]))

    print("→ tool_max_rounds=1: the decider is asked once, then the talker answers...")
    once = TWO_MODEL.model_copy(update={"tool_max_rounds": 1})
    decider, talker = FakeDecider([0.9, 0.9, 0.9]), FakeTalker()
    resp = await agent.run(once, talker, toolset, [{"role": "user", "content": "add 2 and 3"}], {}, tool_backend=decider)
    assert resp["choices"][0]["message"]["content"] == "Result: 5.0", resp
    assert len(decider.seen) == 1 and len(resp["x_aiproxy"]["decisions"]) == 1
    print("  one decision, tool ran, talker answered")

    print("→ tool_fallback: a declined first decision still runs the configured call...")
    from app.config import ToolFallbackConfig

    fb = TWO_MODEL.model_copy(
        update={
            "tool_max_rounds": 1,
            "tool_fallback": ToolFallbackConfig(tool="echo__uppercase", arguments={"text": "{user}"}),
        }
    )
    decider, talker = FakeDecider([None]), FakeTalker()
    resp = await agent.run(fb, talker, toolset, [{"role": "user", "content": "shout"}], {}, tool_backend=decider)
    assert resp["choices"][0]["message"]["content"] == "Result: SHOUT", resp
    d = resp["x_aiproxy"]["decisions"][0]
    assert d["executed"] is False and d["fallback"] == [{"name": "echo__uppercase", "arguments": '{"text": "shout"}'}], d
    print("  fallback ran echo__uppercase('shout'), talker answered from it")

    print("→ two-model assistant: low confidence skips the tool...")
    decider, talker = FakeDecider([0.2]), FakeTalker()
    resp = await agent.run(TWO_MODEL, talker, toolset, [{"role": "user", "content": "hi"}], {}, tool_backend=decider)
    assert resp["choices"][0]["message"]["content"] == "No tool was used.", resp
    assert resp["x_aiproxy"]["decisions"][0]["executed"] is False
    print("  talker answered without tools")

    print("→ two-model assistant: streaming...")
    decider, talker = FakeDecider([0.95, None]), FakeTalker()
    lines = []
    async for line in agent.run_stream(
        TWO_MODEL, talker, toolset, [{"role": "user", "content": "add 2 and 3"}], {}, tool_backend=decider
    ):
        lines.append(line)
    text = "".join(lines)
    streamed, final_obj = "", None
    for line in text.splitlines():
        if line.startswith("data:") and "[DONE]" not in line:
            obj = json.loads(line[5:].strip())
            streamed += obj["choices"][0]["delta"].get("content", "")
            if obj["choices"][0].get("finish_reason"):
                final_obj = obj
    assert streamed.strip() == "Result: 5.0", repr(streamed)
    assert final_obj and [d["executed"] for d in final_obj["x_aiproxy"]["decisions"]] == [True, False]
    print(f"  streamed: {streamed.strip()!r}")

    await mgr.shutdown()
    print("\n✅ ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
