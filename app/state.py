"""Shared, mutable application state.

Holds the live registry (assistants, backends, MCP manager) built from config and
mutated at runtime by the admin API. Backend clients are created lazily and cached
per backend name; replacing a backend disposes the old client.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from .auth import AuthManager
from .backends import Backend, build_backend
from .config import AppConfig, AssistantConfig, BackendConfig, MCPServerConfig, load_config
from .mcp_manager import MCPManager


class AppState:
    def __init__(self, config: AppConfig):
        self.config = config
        self.mcp = MCPManager(config.mcp_servers)
        self.assistants: dict[str, AssistantConfig] = {a.name: a for a in config.assistants}
        self.auth = AuthManager.from_config(config)
        self._backends: dict[str, Backend] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> "AppState":
        return cls(load_config())

    # --- lookups -----------------------------------------------------------
    def get_assistant(self, name: str) -> Optional[AssistantConfig]:
        return self.assistants.get(name)

    def backend_for(self, assistant: AssistantConfig) -> Backend:
        name = assistant.backend
        if name not in self.config.backends:
            raise KeyError(f"assistant '{assistant.name}' references unknown backend '{name}'")
        if name not in self._backends:
            self._backends[name] = build_backend(self.config.backends[name])
        return self._backends[name]

    # --- admin mutations ---------------------------------------------------
    async def upsert_assistant(self, cfg: AssistantConfig) -> None:
        async with self._lock:
            self.assistants[cfg.name] = cfg
            self.config.assistants = [a for a in self.config.assistants if a.name != cfg.name]
            self.config.assistants.append(cfg)

    async def delete_assistant(self, name: str) -> bool:
        async with self._lock:
            existed = self.assistants.pop(name, None) is not None
            self.config.assistants = [a for a in self.config.assistants if a.name != name]
            return existed

    async def upsert_backend(self, name: str, cfg: BackendConfig) -> None:
        async with self._lock:
            self.config.backends[name] = cfg
            old = self._backends.pop(name, None)
        if old is not None:
            await old.aclose()

    async def delete_backend(self, name: str) -> bool:
        async with self._lock:
            existed = self.config.backends.pop(name, None) is not None
            old = self._backends.pop(name, None)
        if old is not None:
            await old.aclose()
        return existed

    async def upsert_mcp_server(self, name: str, cfg: MCPServerConfig) -> None:
        async with self._lock:
            self.config.mcp_servers[name] = cfg
            self.mcp.add_or_replace(name, cfg)

    async def delete_mcp_server(self, name: str) -> bool:
        async with self._lock:
            existed = self.config.mcp_servers.pop(name, None) is not None
        if existed:
            await self.mcp.remove(name)
        return existed

    async def shutdown(self) -> None:
        await self.mcp.shutdown()
        for backend in list(self._backends.values()):
            await backend.aclose()
        await self.auth.aclose()
