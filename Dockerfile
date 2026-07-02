# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Node (for `npx` MCP servers) and CA certs. uv (below) provides `uvx` for
# Python MCP servers. curl kept for healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# uv + uvx, used to install deps and to run uvx-based MCP servers at runtime.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY app ./app
RUN uv pip install --system --no-cache .

# Application code (examples/scripts useful in-container).
COPY examples ./examples
COPY scripts ./scripts

ENV CONFIG_PATH=/app/config.yaml \
    LOG_LEVEL=INFO \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
