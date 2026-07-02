"""End-to-end smoke test with NO real LLM required.

Exercises the real MCP manager (spawning the stdio echo server), the ToolSet
namespacing/routing, and the full agent loop (both non-streaming and streaming)
against a scripted fake backend that asks for a tool then answers.

Run:  python scripts/smoke_test.py
Exits non-zero on failure.
"""

import asyncio
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
    import json

    streamed = ""
    for line in text.splitlines():
        if line.startswith("data:") and "[DONE]" not in line:
            delta = json.loads(line[5:].strip())["choices"][0]["delta"]
            streamed += delta.get("content", "")
    assert streamed == "The sum is 5.", repr(streamed)
    print(f"  streamed content: {streamed!r}")

    await mgr.shutdown()
    print("\n✅ ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
