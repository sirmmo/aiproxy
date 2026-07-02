"""Admin API: inspect and mutate the live registry at runtime.

Changes are in-memory (not written back to ``config.yaml``); ``GET /admin/config``
returns the current state — with secrets redacted — so it can be persisted by hand.
Protected by ``ADMIN_API_KEY`` when that env var is set.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth import require_admin_auth
from ..config import AssistantConfig, BackendConfig, MCPServerConfig
from ..state import AppState

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin_auth)])


def _state(request: Request) -> AppState:
    return request.app.state.app_state


def _redact(cfg: BackendConfig) -> dict[str, Any]:
    data = cfg.model_dump()
    if data.get("api_key"):
        data["api_key"] = "***redacted***"
    return data


@router.get("/config")
async def dump_config(request: Request) -> dict[str, Any]:
    state = _state(request)
    return {
        "mcp_servers": {n: c.model_dump() for n, c in state.config.mcp_servers.items()},
        "backends": {n: _redact(c) for n, c in state.config.backends.items()},
        "assistants": [a.model_dump() for a in state.assistants.values()],
        "proxy_api_keys": ["***redacted***"] if state.config.proxy_api_keys else [],
    }


# --- assistants ------------------------------------------------------------
@router.get("/assistants")
async def list_assistants(request: Request) -> list[dict[str, Any]]:
    return [a.model_dump() for a in _state(request).assistants.values()]


@router.put("/assistants/{name}")
async def put_assistant(name: str, request: Request) -> dict[str, Any]:
    body = await request.json()
    body["name"] = name
    try:
        cfg = AssistantConfig(**body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    state = _state(request)
    if cfg.backend not in state.config.backends:
        raise HTTPException(status_code=422, detail=f"unknown backend '{cfg.backend}'")
    unknown = [s for s in cfg.mcp_servers if s not in state.config.mcp_servers]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown MCP server(s): {unknown}")
    await state.upsert_assistant(cfg)
    return cfg.model_dump()


@router.delete("/assistants/{name}")
async def delete_assistant(name: str, request: Request) -> dict[str, str]:
    if not await _state(request).delete_assistant(name):
        raise HTTPException(status_code=404, detail=f"assistant '{name}' not found")
    return {"status": "deleted", "name": name}


# --- backends --------------------------------------------------------------
@router.get("/backends")
async def list_backends(request: Request) -> dict[str, Any]:
    return {n: _redact(c) for n, c in _state(request).config.backends.items()}


@router.put("/backends/{name}")
async def put_backend(name: str, request: Request) -> dict[str, Any]:
    body = await request.json()
    try:
        cfg = BackendConfig(**body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    await _state(request).upsert_backend(name, cfg)
    return {"name": name, **_redact(cfg)}


@router.delete("/backends/{name}")
async def delete_backend(name: str, request: Request) -> dict[str, str]:
    if not await _state(request).delete_backend(name):
        raise HTTPException(status_code=404, detail=f"backend '{name}' not found")
    return {"status": "deleted", "name": name}


# --- MCP servers -----------------------------------------------------------
@router.get("/mcp")
async def list_mcp(request: Request) -> dict[str, Any]:
    return {n: c.model_dump() for n, c in _state(request).config.mcp_servers.items()}


@router.put("/mcp/{name}")
async def put_mcp(name: str, request: Request) -> dict[str, Any]:
    body = await request.json()
    try:
        cfg = MCPServerConfig(**body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    await _state(request).upsert_mcp_server(name, cfg)
    return {"name": name, **cfg.model_dump()}


@router.delete("/mcp/{name}")
async def delete_mcp(name: str, request: Request) -> dict[str, str]:
    if not await _state(request).delete_mcp_server(name):
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' not found")
    return {"status": "deleted", "name": name}


@router.get("/mcp/{name}/tools")
async def mcp_tools(name: str, request: Request) -> dict[str, Any]:
    """Connect (if needed) and return the tools a server advertises — handy for debugging."""
    state = _state(request)
    if name not in state.mcp.servers:
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' not found")
    try:
        await state.mcp.ensure_started([name])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"failed to start '{name}': {exc}")
    return {"name": name, "tools": state.mcp.servers[name].tools}
