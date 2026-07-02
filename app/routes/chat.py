"""OpenAI-compatible surface: /v1/models and /v1/chat/completions."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import agent
from ..auth import require_proxy_auth
from ..backends.base import SAMPLING_KEYS
from ..state import AppState

logger = logging.getLogger("aiproxy.chat")

router = APIRouter(dependencies=[Depends(require_proxy_auth)])


def _state(request: Request) -> AppState:
    return request.app.state.app_state


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    state = _state(request)
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": a.name,
                "object": "model",
                "created": now,
                "owned_by": "aiproxy",
                "description": a.description,
                "backend": a.backend,
                "mcp_servers": a.mcp_servers,
            }
            for a in state.assistants.values()
        ],
    }


def _extract_params(body: dict[str, Any], assistant) -> dict[str, Any]:
    """Merge assistant defaults with client-supplied sampling params (client wins)."""
    params: dict[str, Any] = {}
    for key in ("temperature", "top_p", "max_tokens"):
        default = getattr(assistant, key, None)
        if default is not None:
            params[key] = default
    for key in SAMPLING_KEYS:
        if body.get(key) is not None:
            params[key] = body[key]
    if assistant.extra_body:
        params["extra_body"] = dict(assistant.extra_body)
    return params


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    state = _state(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="request body must be JSON")

    model = body.get("model")
    messages = body.get("messages")
    if not model or not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="'model' and 'messages' are required")

    assistant = state.get_assistant(model)
    if assistant is None:
        raise HTTPException(
            status_code=404,
            detail=f"model '{model}' not found. See GET /v1/models for available assistants.",
        )

    try:
        backend = state.backend_for(assistant)
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        await state.mcp.ensure_started(assistant.mcp_servers)
    except Exception as exc:  # noqa: BLE001 - MCP connect/list failure
        logger.exception("failed to start MCP servers for %s", assistant.name)
        raise HTTPException(status_code=502, detail=f"MCP server error: {exc}")

    toolset = state.mcp.build_toolset(assistant.mcp_servers)
    params = _extract_params(body, assistant)
    stream = bool(body.get("stream"))

    if stream:
        return StreamingResponse(
            agent.run_stream(assistant, backend, toolset, messages, params),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await agent.run(assistant, backend, toolset, messages, params)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        status = exc.response.status_code if exc.response is not None else 502
        raise HTTPException(status_code=status, detail=f"upstream LLM error: {detail}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("completion failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(result)
