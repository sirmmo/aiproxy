"""Unit tests for the pluggable auth chain (static keys + Apiman modes).

Uses httpx.MockTransport to simulate the Apiman gateway, so no network is needed.

Run:  python scripts/test_auth.py   (exits non-zero on failure)
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from starlette.requests import Request  # noqa: E402

from app.auth import (  # noqa: E402
    ApimanGatewayProbeProvider,
    ApimanTrustedHeaderProvider,
    AuthManager,
    StaticKeyProvider,
    extract_credential,
)
from app.config import ApimanConfig  # noqa: E402


def make_request(headers: dict | None = None, query: str = "") -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": raw,
        "query_string": query.encode(),
    }
    return Request(scope)


async def authorized(manager: AuthManager, request: Request) -> bool:
    try:
        await manager.authorize(request)
        return True
    except HTTPException as exc:
        assert exc.status_code == 401, exc.status_code
        return False


async def main() -> None:
    # --- credential extraction (bearer, X-API-Key, apikey query; de-duped) ---
    cred = extract_credential(
        make_request({"Authorization": "Bearer k1", "X-API-Key": "k2"}, "apikey=k3")
    )
    assert cred.candidate_keys == ["k1", "k2", "k3"], cred.candidate_keys
    cred = extract_credential(make_request({"Authorization": "Bearer dup", "X-API-Key": "dup"}))
    assert cred.candidate_keys == ["dup"], cred.candidate_keys
    print("✓ credential extraction (bearer / X-API-Key / apikey, de-duped)")

    # --- static keys via any carrier ---
    static = AuthManager([StaticKeyProvider({"sk-good"})])
    assert await authorized(static, make_request({"Authorization": "Bearer sk-good"}))
    assert await authorized(static, make_request({"X-API-Key": "sk-good"}))
    assert await authorized(static, make_request(query="apikey=sk-good"))
    assert not await authorized(static, make_request({"Authorization": "Bearer nope"}))
    assert not await authorized(static, make_request())
    print("✓ static keys accept valid key via header/bearer/query, reject others")

    # --- apiman trusted_header ---
    th = AuthManager([ApimanTrustedHeaderProvider("X-Gw-Token", "s3cret")])
    assert await authorized(th, make_request({"X-Gw-Token": "s3cret"}))
    assert not await authorized(th, make_request({"X-Gw-Token": "wrong"}))
    assert not await authorized(th, make_request())
    print("✓ apiman trusted_header accepts matching secret, rejects otherwise")

    # --- apiman gateway_probe (MockTransport gateway) + caching ---
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        key = request.headers.get("X-API-Key")
        assert str(request.url) == "http://gw/apiman-gateway/aiproxy/authcheck/1.0/health", request.url
        if key == "good":
            return httpx.Response(200, text="ok")
        if key == "boom":
            return httpx.Response(500, text="misconfigured backend")
        return httpx.Response(401, text="unauthorized")

    cfg = ApimanConfig(
        enabled=True,
        mode="gateway_probe",
        gateway_url="http://gw/apiman-gateway",
        probe_api="aiproxy/authcheck/1.0",
        probe_path="health",
        cache_ttl=60,
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    probe = ApimanGatewayProbeProvider(cfg, client=client)
    gw = AuthManager([probe])

    assert await authorized(gw, make_request({"X-API-Key": "good"}))
    assert not await authorized(gw, make_request({"X-API-Key": "bad"}))
    assert not await authorized(gw, make_request({"X-API-Key": "boom"}))  # 500 -> fail closed
    n_after = calls["n"]
    # Repeat the good key: must be served from cache (no new gateway call).
    assert await authorized(gw, make_request({"Authorization": "Bearer good"}))
    assert calls["n"] == n_after, f"expected cache hit, got {calls['n']} vs {n_after}"
    await client.aclose()
    print("✓ apiman gateway_probe: 2xx=valid, 401/500=reject, results cached")

    # --- parallel: static + apiman both accepted, in one chain ---
    both = AuthManager(
        [StaticKeyProvider({"sk-good"}), ApimanTrustedHeaderProvider("X-Gw-Token", "s3cret")]
    )
    assert await authorized(both, make_request({"Authorization": "Bearer sk-good"}))  # static
    assert await authorized(both, make_request({"X-Gw-Token": "s3cret"}))  # apiman
    assert not await authorized(both, make_request({"Authorization": "Bearer nope"}))
    print("✓ chain authorizes if EITHER static OR apiman accepts (parallel)")

    # --- open when nothing configured ---
    open_mgr = AuthManager([])
    assert not open_mgr.enabled
    assert await authorized(open_mgr, make_request())
    print("✓ auth is open when no providers configured")

    # --- config validation guards misconfiguration ---
    for bad in (
        dict(enabled=True, mode="gateway_probe"),  # missing gateway_url/probe_api
        dict(enabled=True, mode="gateway_probe", gateway_url="x", probe_api="only/two"),
        dict(enabled=True, mode="trusted_header"),  # missing header/secret
    ):
        try:
            ApimanConfig(**bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    print("✓ apiman config validation rejects incomplete settings")

    print("\n✅ ALL AUTH CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
