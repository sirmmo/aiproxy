"""Shared judge-model client for the LLM-judged scorers. Any OpenAI-compatible endpoint:
    JUDGE_BASE_URL=https://openrouter.ai/api/v1  JUDGE_API_KEY=...  JUDGE_MODEL=deepseek/deepseek-chat"""
import os, sys
from openai import OpenAI

BASE = os.environ.get("JUDGE_BASE_URL", "")
KEY = os.environ.get("JUDGE_API_KEY", "")
MODEL = os.environ.get("JUDGE_MODEL", "")

def require():
    if not (BASE and MODEL):
        sys.exit("set JUDGE_BASE_URL, JUDGE_API_KEY and JUDGE_MODEL (an OpenAI-compatible judge endpoint)")

def client() -> OpenAI:
    require()
    return OpenAI(base_url=BASE, api_key=KEY or "not-needed", timeout=180, max_retries=3)

def ask(cl: OpenAI, prompt: str, max_tokens: int = 200, system: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    r = cl.chat.completions.create(model=MODEL, messages=msgs, temperature=0, max_tokens=max_tokens)
    return (r.choices[0].message.content or "").strip()

def load_records(path: str):
    import json
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if not r.get("error")]
