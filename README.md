# aiproxy

[![ci](https://github.com/sirmmo/aiproxy/actions/workflows/ci.yml/badge.svg)](https://github.com/sirmmo/aiproxy/actions/workflows/ci.yml)
[![publish](https://github.com/sirmmo/aiproxy/actions/workflows/publish.yml/badge.svg)](https://github.com/sirmmo/aiproxy/actions/workflows/publish.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ghcr.io](https://img.shields.io/badge/ghcr.io-sirmmo%2Faiproxy-2496ED?logo=docker&logoColor=white)](https://github.com/sirmmo/aiproxy/pkgs/container/aiproxy)

**An OpenAI-compatible gateway that turns any LLM into a more capable one by fusing it with a reusable fabric of MCP servers.**

📖 **Documentation:** https://sirmmo.github.io/aiproxy/

Point any OpenAI client at aiproxy, pick one of your configured *assistants* as the `model`, and the gateway runs the whole agentic tool loop for you — calling the wrapped LLM, executing tools against your [Model Context Protocol](https://modelcontextprotocol.io) servers, feeding results back — and returns a normal OpenAI response. From the client's side it just looks like a smarter model.

```
┌────────────┐   OpenAI /v1/chat/completions   ┌──────────────────────────────┐
│ Any OpenAI │ ──────────────────────────────► │            aiproxy           │
│   client   │ ◄────────────────────────────── │  ┌────────────────────────┐  │
└────────────┘        OpenAI response          │  │      agent loop        │  │
                                                │  └───┬───────────────┬────┘  │
                                                │      │ LLM turn      │ tools │
                                                │      ▼               ▼       │
                                                │  ┌────────┐   ┌────────────┐ │
                                                │  │ backend│   │ MCP servers│ │
                                                │  │ OpenAI │   │ fetch, fs, │ │
                                                │  │/Anthro-│   │ http, ...  │ │
                                                │  │  pic   │   └────────────┘ │
                                                │  └───┬────┘                  │
                                                └──────┼───────────────────────┘
                                                       ▼
                                          OpenAI / Anthropic / local model
```

## Why

MCP gives you a growing ecosystem of tool servers (web fetch, filesystem, databases, search, your own APIs). But wiring those tools into *every* app and *every* model is repetitive. aiproxy makes that infrastructure **reusable and model-agnostic**: define your MCP servers and LLM backends once, compose them into named assistants, and every OpenAI-compatible app in your stack gets a tool-augmented model for free — no client changes, no SDK lock-in.

## Features

- **OpenAI-compatible API** — `/v1/chat/completions` (streaming *and* non-streaming) and `/v1/models`. Works with the OpenAI SDKs, LangChain, LlamaIndex, `curl`, etc.
- **Wraps any LLM** — OpenAI-compatible backends (OpenAI, Groq, Together, Mistral, vLLM, Ollama, LM Studio, OpenRouter…) **and** native Anthropic (Messages API), behind one interface.
- **Reusable MCP fabric** — attach any number of MCP servers (`stdio`, `sse`, `streamable-http`) to each assistant. Tools are namespaced per server and executed transparently.
- **Assistants as virtual models** — each `model` a client can pick is a backend + system prompt + set of MCP servers + tool-loop budget.
- **Runtime admin API** — add/edit/remove assistants, backends and MCP servers without a restart; introspect any server's tools.
- **Docker-first** — `docker compose up` and you have an endpoint. Node (`npx`) and `uvx` are baked in so most MCP servers install on demand.

## Quick start (Docker)

Use the prebuilt image from GitHub Container Registry:

```bash
cp .env.example .env                # add your OPENAI_API_KEY / ANTHROPIC_API_KEY
cp config.example.yaml config.yaml  # define backends, MCP servers, assistants

docker run --rm -p 8000:8000 --env-file .env \
  -v "$PWD/config.yaml:/app/config.yaml:ro" \
  ghcr.io/sirmmo/aiproxy:latest
```

…or build from source with compose:

```bash
cp .env.example .env
cp config.example.yaml config.yaml
docker compose up --build
```

Then talk to it like OpenAI:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "research-assistant",
    "messages": [{"role": "user", "content": "Summarize https://modelcontextprotocol.io"}]
  }'
```

Or with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused-or-your-PROXY_API_KEY")

resp = client.chat.completions.create(
    model="research-assistant",              # an assistant, not a raw model
    messages=[{"role": "user", "content": "What's on the MCP homepage?"}],
)
print(resp.choices[0].message.content)
```

Streaming works exactly as clients expect (`stream=True`) — content tokens flow through while tool rounds run transparently between them.

## Configuration

Everything is declared in `config.yaml`. `${VAR}` / `${VAR:-default}` are expanded from the environment, so keep secrets in `.env`.

```yaml
mcp_servers:
  fetch:                                   # a reusable MCP server
    transport: stdio
    command: uvx
    args: ["mcp-server-fetch"]

backends:
  anthropic:                               # a wrapped LLM provider
    kind: anthropic                        # or "openai" for any compat endpoint
    base_url: https://api.anthropic.com/v1
    api_key: ${ANTHROPIC_API_KEY}

assistants:
  - name: research-assistant               # <- clients pass this as `model`
    backend: anthropic
    model: claude-sonnet-5
    system_prompt: "You are a meticulous research assistant. Cite your sources."
    mcp_servers: [fetch]
    max_tool_iterations: 8
    temperature: 0.2
```

See [`config.example.yaml`](config.example.yaml) for the fully annotated version, including `sse`/`http` MCP servers, local model backends, and API-key auth.

### Backends

| `kind`      | Talks to                          | Auth header            |
|-------------|-----------------------------------|------------------------|
| `openai`    | any OpenAI-compatible `/chat/completions` | `Authorization: Bearer` |
| `anthropic` | native Anthropic `/messages`      | `x-api-key`            |

The Anthropic backend translates the canonical chat messages ↔ the Messages API (system prompt, `tool_use`/`tool_result` blocks, streaming events, stop-reason mapping), so tool use works first-class with Claude.

### MCP servers

| `transport`               | Fields                         |
|---------------------------|--------------------------------|
| `stdio`                   | `command`, `args`, `env`, `cwd`|
| `sse`                     | `url`, `headers`               |
| `http` / `streamable-http`| `url`, `headers`               |

Tools are exposed to the model as `"<server>__<tool>"` and routed back to the right server on call. Sessions are persistent (one subprocess per stdio server, reused across requests) and started lazily on first use.

## Admin API

Mutate the live registry without restarting (set `ADMIN_API_KEY` to protect it):

```bash
# Inspect current state (secrets redacted)
curl localhost:8000/admin/config

# See what tools a server actually advertises
curl localhost:8000/admin/mcp/fetch/tools

# Add/replace an assistant
curl -X PUT localhost:8000/admin/assistants/coder \
  -H "Content-Type: application/json" \
  -d '{"backend":"openai","model":"gpt-4o","mcp_servers":["filesystem"],
       "system_prompt":"You are a coding agent."}'
```

| Method & path | Purpose |
|---|---|
| `GET /admin/config` | Dump current registry (secrets redacted) |
| `GET/PUT/DELETE /admin/assistants[/{name}]` | Manage assistants |
| `GET/PUT/DELETE /admin/backends[/{name}]` | Manage LLM backends |
| `GET/PUT/DELETE /admin/mcp[/{name}]` | Manage MCP servers |
| `GET /admin/mcp/{name}/tools` | Introspect a server's tools |

> Admin changes are in-memory. Use `GET /admin/config` to export current state and persist it into `config.yaml` yourself.

## Auth

- **`/v1/*`** — if `proxy_api_keys` is non-empty in config, clients must send `Authorization: Bearer <key>`.
- **`/admin/*`** — if `ADMIN_API_KEY` is set, admin calls must send `Authorization: Bearer <ADMIN_API_KEY>`.

Both are open when unset (handy for local dev; lock them down in production).

## Development & testing

The recommended path is Docker — it pins a clean Python 3.12 with `node`/`uvx`
available, so you never fight a host interpreter or a leaked `PYTHONPATH`.

```bash
docker build -t aiproxy:latest .

# End-to-end check — spawns the demo MCP server and drives the full agent
# loop (streaming + non-streaming) with a scripted fake LLM. No API key needed:
docker run --rm aiproxy:latest python scripts/smoke_test.py

# Run the server against your config.yaml (or just `docker compose up`):
docker run --rm -p 8000:8000 -v "$PWD/config.yaml:/app/config.yaml:ro" aiproxy:latest
```

<details>
<summary>Running without Docker (uv venv)</summary>

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
# Clear PYTHONPATH if your shell injects system site-packages that shadow the venv:
env PYTHONPATH= .venv/bin/python scripts/smoke_test.py
env PYTHONPATH= .venv/bin/uvicorn app.main:app --reload --port 8000
```
</details>

## How a request flows

1. Client `POST /v1/chat/completions` with `model: "<assistant>"`.
2. Gateway resolves the assistant → backend + MCP servers, and ensures those servers are connected.
3. It builds the OpenAI tool schema from the servers' tools and enters the agent loop:
   - call the backend LLM with the messages + tools;
   - if the model returns tool calls, execute them **concurrently** against the MCP servers and append the results;
   - repeat until the model answers or `max_tool_iterations` is hit (the last turn drops tools to force a final answer).
4. Return an OpenAI `chat.completion` (or stream `chat.completion.chunk`s), with the assistant name as the `model` and aggregated token usage.

## Project layout

```
app/
  main.py            FastAPI app + lifespan
  config.py          YAML config models, ${env} expansion
  state.py           live registry (assistants/backends/MCP), runtime mutation
  mcp_manager.py     persistent MCP sessions + namespaced ToolSet router
  agent.py           the agentic loop (streaming + non-streaming)
  backends/          openai + anthropic adapters behind one interface
  routes/            /v1 (chat) and /admin (registry) endpoints
  auth.py            bearer-token auth
examples/echo_mcp_server.py   demo stdio MCP server
scripts/smoke_test.py         end-to-end test, no real LLM required
```

## License

MIT
