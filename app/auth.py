"""Auth for the public (/v1) surface — a pluggable chain of providers.

A request is authorized if **any** enabled provider accepts it, so static keys
(``proxy_api_keys``) and Apiman validation run in parallel. If no provider is
configured the surface is open (handy for local dev).

Providers:
* :class:`StaticKeyProvider`          – key ∈ configured ``proxy_api_keys``.
* :class:`ApimanGatewayProbeProvider` – key validated via a round-trip through
                                        the Apiman gateway (2xx = valid), cached.
* :class:`ApimanTrustedHeaderProvider`– request carries the shared secret the
                                        Apiman gateway injects in front of us.

Admin (/admin) auth stays separate: a single ``ADMIN_API_KEY`` env var.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

import httpx
from fastapi import HTTPException, Request, status

from .config import ApimanConfig, AppConfig

logger = logging.getLogger("aiproxy.auth")


# --------------------------------------------------------------------------- #
# Credential extraction
# --------------------------------------------------------------------------- #
def _bearer(request: Request) -> Optional[str]:
    header = request.headers.get("authorization")
    if not header:
        return None
    parts = header.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return header.strip()


@dataclass
class Credential:
    """All the places a caller's API key might live, de-duplicated and ordered."""

    candidate_keys: list[str] = field(default_factory=list)


def extract_credential(request: Request) -> Credential:
    seen: set[str] = set()
    keys: list[str] = []
    for value in (
        _bearer(request),
        request.headers.get("x-api-key"),
        request.query_params.get("apikey"),
    ):
        if value and value not in seen:
            seen.add(value)
            keys.append(value)
    return Credential(candidate_keys=keys)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class AuthProvider(Protocol):
    name: str

    async def authorize(self, request: Request, cred: Credential) -> bool: ...

    async def aclose(self) -> None: ...


class StaticKeyProvider:
    name = "static"

    def __init__(self, keys: set[str]):
        self._keys = keys

    async def authorize(self, request: Request, cred: Credential) -> bool:
        # Constant-time compare against each configured key.
        for candidate in cred.candidate_keys:
            for key in self._keys:
                if hmac.compare_digest(candidate, key):
                    return True
        return False

    async def aclose(self) -> None:
        pass


class ApimanTrustedHeaderProvider:
    """Trust requests forwarded by an Apiman gateway that injects a shared secret."""

    name = "apiman:trusted_header"

    def __init__(self, header: str, secret: str):
        self._header = header
        self._secret = secret

    async def authorize(self, request: Request, cred: Credential) -> bool:
        value = request.headers.get(self._header)
        return bool(value) and hmac.compare_digest(value, self._secret)

    async def aclose(self) -> None:
        pass


class ApimanGatewayProbeProvider:
    """Validate a key by round-tripping through the Apiman gateway to a probe API."""

    name = "apiman:gateway_probe"

    def __init__(self, cfg: ApimanConfig, client: Optional[httpx.AsyncClient] = None):
        base = cfg.gateway_url.rstrip("/") + "/" + cfg.probe_api.strip("/")
        self._url = base + "/" + cfg.probe_path.lstrip("/") if cfg.probe_path else base + "/"
        self._api_key_header = cfg.api_key_header
        self._ttl = cfg.cache_ttl
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(cfg.timeout, connect=5.0))
        self._cache: dict[str, tuple[bool, float]] = {}

    async def authorize(self, request: Request, cred: Credential) -> bool:
        for key in cred.candidate_keys:
            if await self._check(key):
                return True
        return False

    async def _check(self, key: str) -> bool:
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached is not None and cached[1] > now:
            return cached[0]
        authorized = await self._probe(key)
        self._cache[key] = (authorized, now + self._ttl)
        self._prune(now)
        return authorized

    async def _probe(self, key: str) -> bool:
        try:
            resp = await self._client.get(self._url, headers={self._api_key_header: key})
        except httpx.HTTPError as exc:
            # Fail closed on transport errors, but don't wedge the request path.
            logger.warning("apiman probe failed (%s): %s", self._url, exc)
            return False
        if 200 <= resp.status_code < 300:
            return True
        if resp.status_code in (401, 403):
            return False
        # A valid key proxies to the backend; a non-2xx backend response here means
        # the probe API is misconfigured (should map to an always-2xx endpoint).
        logger.warning(
            "apiman probe returned unexpected status %s for %s; treating as unauthorized",
            resp.status_code,
            self._url,
        )
        return False

    def _prune(self, now: float) -> None:
        if len(self._cache) > 2048:
            self._cache = {k: v for k, v in self._cache.items() if v[1] > now}

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #
class AuthManager:
    def __init__(self, providers: list[AuthProvider]):
        self.providers = providers
        self.enabled = bool(providers)

    @classmethod
    def from_config(cls, config: AppConfig) -> "AuthManager":
        providers: list[AuthProvider] = []
        if config.proxy_api_keys:
            providers.append(StaticKeyProvider(set(config.proxy_api_keys)))
        ap = config.apiman
        if ap.enabled:
            if ap.mode == "trusted_header":
                providers.append(ApimanTrustedHeaderProvider(ap.header, ap.secret))
            else:
                providers.append(ApimanGatewayProbeProvider(ap))
        if providers:
            logger.info("auth enabled with providers: %s", [p.name for p in providers])
        else:
            logger.info("auth is OPEN (no proxy_api_keys and apiman disabled)")
        return cls(providers)

    async def authorize(self, request: Request) -> None:
        """Raise 401 unless some provider accepts the request; no-op if disabled."""
        if not self.enabled:
            return
        cred = extract_credential(request)
        for provider in self.providers:
            try:
                if await provider.authorize(request, cred):
                    return
            except Exception:  # noqa: BLE001 - a broken provider must not 500 the request
                logger.exception("auth provider '%s' errored", provider.name)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    async def aclose(self) -> None:
        for provider in self.providers:
            await provider.aclose()


# --------------------------------------------------------------------------- #
# FastAPI dependencies
# --------------------------------------------------------------------------- #
async def require_proxy_auth(request: Request) -> None:
    await request.app.state.app_state.auth.authorize(request)


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
