"""Bearer-token auth for the public (/v1) and admin (/admin) surfaces."""

from __future__ import annotations

import os

from fastapi import HTTPException, Request, status


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not header:
        return None
    parts = header.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return header.strip()


def require_proxy_auth(request: Request) -> None:
    """Enforce ``proxy_api_keys`` from config on the OpenAI-compatible endpoints."""
    keys = request.app.state.app_state.config.proxy_api_keys
    if not keys:
        return
    token = _bearer(request)
    if token not in keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_admin_auth(request: Request) -> None:
    """Enforce the ``ADMIN_API_KEY`` env var on the admin endpoints (open if unset)."""
    admin_key = os.environ.get("ADMIN_API_KEY")
    if not admin_key:
        return
    if _bearer(request) != admin_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing admin key",
            headers={"WWW-Authenticate": "Bearer"},
        )
