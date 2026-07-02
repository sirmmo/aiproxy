"""FastAPI application entrypoint.

Run with:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .routes import admin, chat
from .state import AppState

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("aiproxy")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState.from_env()
    app.state.app_state = state
    logger.info(
        "loaded %d assistant(s), %d backend(s), %d MCP server(s)",
        len(state.assistants),
        len(state.config.backends),
        len(state.mcp.servers),
    )
    try:
        yield
    finally:
        await state.shutdown()
        logger.info("shutdown complete")


app = FastAPI(
    title="aiproxy",
    description="OpenAI-compatible gateway that augments any LLM with a reusable fabric of MCP servers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "name": "aiproxy",
            "openai_base_url": "/v1",
            "endpoints": {
                "models": "GET /v1/models",
                "chat": "POST /v1/chat/completions",
                "admin": "GET /admin/config",
                "health": "GET /health",
            },
        }
    )
