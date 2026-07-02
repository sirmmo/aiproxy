"""Manages long-lived MCP client sessions and exposes their tools to the agent loop.

Each MCP server runs inside its own asyncio task that owns the connection's
``async with`` scope for its entire lifetime. Tool calls are handed to that task
via a queue and answered through a future. This respects the anyio cancel-scope
rules of the MCP SDK (a session must be entered and exited from the same task)
while still allowing many concurrent requests to share one persistent session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import AsyncExitStack
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import MCPServerConfig

logger = logging.getLogger("aiproxy.mcp")


def _stringify_result(result: Any) -> str:
    """Flatten an MCP CallToolResult into text the LLM can read."""
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
            continue
        data = getattr(block, "data", None)
        if data is not None:  # image / blob
            mime = getattr(block, "mimeType", "application/octet-stream")
            parts.append(f"[binary {mime} content omitted]")
            continue
        dump = block.model_dump() if hasattr(block, "model_dump") else block
        parts.append(json.dumps(dump, default=str))
    text = "\n".join(parts).strip()
    if getattr(result, "isError", False):
        return f"[tool error] {text}" if text else "[tool error]"
    return text or "(no output)"


class MCPServer:
    """A single connected MCP server."""

    def __init__(self, name: str, cfg: MCPServerConfig):
        self.name = name
        self.cfg = cfg
        self.tools: list[dict[str, Any]] = []  # normalized: {name, description, input_schema}
        self._task: Optional[asyncio.Task] = None
        self._queue: "asyncio.Queue" = asyncio.Queue()
        self._ready = asyncio.Event()
        self._error: Optional[BaseException] = None
        self._stop = asyncio.Event()
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        """Connect and list tools. Idempotent and safe under concurrency."""
        async with self._start_lock:
            if self._task and self._ready.is_set() and self._error is None and not self._task.done():
                return
            if self._task is None or self._task.done():
                self._ready.clear()
                self._error = None
                self._stop.clear()
                self._task = asyncio.create_task(self._run(), name=f"mcp:{self.name}")
            await self._ready.wait()
            if self._error is not None:
                raise RuntimeError(f"MCP server '{self.name}' failed to start: {self._error}")

    async def _connect(self, stack: AsyncExitStack):
        c = self.cfg
        if c.transport == "stdio":
            if not c.command:
                raise ValueError("stdio transport requires 'command'")
            params = StdioServerParameters(
                command=c.command,
                args=list(c.args),
                env={**os.environ, **c.env} if c.env else dict(os.environ),
                cwd=c.cwd,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            return read, write
        if c.transport == "sse":
            from mcp.client.sse import sse_client

            if not c.url:
                raise ValueError("sse transport requires 'url'")
            read, write = await stack.enter_async_context(sse_client(c.url, headers=c.headers or None))
            return read, write
        # http / streamable-http
        from mcp.client.streamable_http import streamablehttp_client

        if not c.url:
            raise ValueError("http transport requires 'url'")
        read, write, _ = await stack.enter_async_context(
            streamablehttp_client(c.url, headers=c.headers or None)
        )
        return read, write

    async def _run(self) -> None:
        try:
            async with AsyncExitStack() as stack:
                read, write = await self._connect(stack)
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed = await session.list_tools()
                self.tools = [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.inputSchema or {"type": "object", "properties": {}},
                    }
                    for t in listed.tools
                ]
                logger.info("MCP '%s' ready with %d tool(s)", self.name, len(self.tools))
                self._ready.set()
                while not self._stop.is_set():
                    item = await self._queue.get()
                    if item is None:
                        break
                    tool_name, arguments, fut = item
                    try:
                        result = await session.call_tool(tool_name, arguments)
                        if not fut.done():
                            fut.set_result(_stringify_result(result))
                    except Exception as exc:  # noqa: BLE001 - surfaced to caller
                        if not fut.done():
                            fut.set_exception(exc)
        except Exception as exc:  # connection / init failure
            self._error = exc
            logger.exception("MCP '%s' crashed", self.name)
            self._ready.set()
            # Fail any queued work.
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if item is not None and not item[2].done():
                    item[2].set_exception(exc)

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if self._error is not None:
            raise RuntimeError(f"MCP server '{self.name}' is not available: {self._error}")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put((tool_name, arguments, fut))
        return await fut

    async def stop(self) -> None:
        self._stop.set()
        await self._queue.put(None)
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


class ToolSet:
    """The tools available to one assistant, namespaced by server, plus a router.

    OpenAI tool names must match ``^[A-Za-z0-9_-]{1,64}$``, so tools are exposed
    as ``<server>__<tool>`` (sanitized/truncated, de-duplicated on collision).
    """

    def __init__(self, manager: "MCPManager", server_names: list[str]):
        self._manager = manager
        self.tools: list[dict[str, Any]] = []  # OpenAI tool schema
        self._route: dict[str, tuple[str, str]] = {}  # exposed name -> (server, tool)
        for server_name in server_names:
            server = manager.servers.get(server_name)
            if server is None:
                continue
            for tool in server.tools:
                exposed = self._register(server_name, tool["name"])
                self.tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": exposed,
                            "description": tool["description"],
                            "parameters": tool["input_schema"],
                        },
                    }
                )

    def _register(self, server_name: str, tool_name: str) -> str:
        base = f"{_sanitize(server_name)}__{_sanitize(tool_name)}"[:64]
        exposed = base
        i = 1
        while exposed in self._route:
            suffix = f"_{i}"
            exposed = base[: 64 - len(suffix)] + suffix
            i += 1
        self._route[exposed] = (server_name, tool_name)
        return exposed

    @property
    def is_empty(self) -> bool:
        return not self.tools

    async def call(self, exposed_name: str, arguments: dict[str, Any]) -> str:
        route = self._route.get(exposed_name)
        if route is None:
            return f"[tool error] unknown tool '{exposed_name}'"
        server_name, tool_name = route
        server = self._manager.servers.get(server_name)
        if server is None:
            return f"[tool error] server '{server_name}' is not registered"
        return await server.call(tool_name, arguments)


class MCPManager:
    """Owns all MCP server sessions; starts them lazily on first use."""

    def __init__(self, servers_cfg: dict[str, MCPServerConfig]):
        self.servers: dict[str, MCPServer] = {
            name: MCPServer(name, cfg) for name, cfg in servers_cfg.items()
        }

    async def ensure_started(self, names: list[str]) -> None:
        missing = [n for n in names if n not in self.servers]
        if missing:
            raise KeyError(f"unknown MCP server(s): {', '.join(missing)}")
        await asyncio.gather(*(self.servers[n].start() for n in names))

    def build_toolset(self, names: list[str]) -> ToolSet:
        return ToolSet(self, names)

    # --- runtime registry mutations (admin API) ---
    def add_or_replace(self, name: str, cfg: MCPServerConfig) -> None:
        old = self.servers.get(name)
        self.servers[name] = MCPServer(name, cfg)
        if old is not None:
            asyncio.create_task(old.stop())

    async def remove(self, name: str) -> None:
        server = self.servers.pop(name, None)
        if server is not None:
            await server.stop()

    async def shutdown(self) -> None:
        await asyncio.gather(*(s.stop() for s in self.servers.values()), return_exceptions=True)
