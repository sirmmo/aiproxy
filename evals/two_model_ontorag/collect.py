"""Run the RAG test set through an aiproxy assistant and record question,
retrieved contexts (from x_aiproxy.decisions[].results) and answer. Resumable."""
import json, os, re, sys, time
from openai import OpenAI

BASE = os.environ.get("AIPROXY_BASE", "http://127.0.0.1:8010/v1")
assistant, testset, out = sys.argv[1], sys.argv[2], sys.argv[3]
only_lang = sys.argv[4] if len(sys.argv) > 4 else None
c = OpenAI(base_url=BASE, api_key="x", timeout=1800)
done = set()
if os.path.exists(out):
    done = {json.loads(l)["qid"] for l in open(out)}
rows = [json.loads(l) for l in open(testset)]
if only_lang:
    rows = [r for r in rows if r["lang"] == only_lang]
CHUNK_ID = re.compile(r"[a-z0-9][a-z0-9\-]+::\d{4}")  # passages cite chunks as book::0123

def contexts_from(results):
    """Split a tool result into context strings; keep chunk ids for recall."""
    ctx, ids = [], []
    for text in results:
        ids += CHUNK_ID.findall(text)
        try:
            obj = json.loads(text.split("\n[truncated:")[0])
        except Exception:
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("passages"), list):
            for p in obj["passages"]:
                if isinstance(p, dict) and p.get("text"):
                    ctx.append(p["text"])
            for f in obj.get("ontology_facts") or []:
                if isinstance(f, dict):
                    ctx.append(f"{f.get('label','')}: {f.get('summary','')}")
        elif isinstance(obj, list):
            for e in obj:
                if isinstance(e, dict):
                    ctx.append(f"{e.get('label','')}: {e.get('summary','')}")
        else:
            ctx.append(text[:4000])
    return [x for x in ctx if x.strip()], sorted(set(ids))

with open(out, "a") as f:
    for r in rows:
        if r["qid"] in done:
            continue
        t = time.time()
        try:
            resp = c.chat.completions.create(model=assistant, messages=[{"role": "user", "content": r["question"]}])
            x = resp.model_extra.get("x_aiproxy") or {}
            decisions = x.get("decisions", [])
            results = [res for d in decisions for res in (d.get("results") or [])]
            ctx, ids = contexts_from(results)
            rec = {**r, "assistant": assistant, "answer": resp.choices[0].message.content, "finish": resp.choices[0].finish_reason,
                   "seconds": round(time.time() - t, 1), "completion_tokens": resp.usage.completion_tokens,
                   "decisions": [{k: v for k, v in d.items() if k != "results"} for d in decisions],
                   "contexts": ctx, "retrieved_ids": ids, "error": None}
        except Exception as e:
            rec = {**r, "assistant": assistant, "answer": "", "finish": "error", "seconds": round(time.time() - t, 1),
                   "completion_tokens": 0, "decisions": [], "contexts": [], "retrieved_ids": [], "error": str(e)[:200]}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(f"[{assistant}] {r['qid']} {rec['seconds']}s finish={rec['finish']} ctx={len(rec['contexts'])} hit={bool(set(rec['retrieved_ids']) & set(r['gold_ids']))} :: {rec['answer'][:80]!r}", flush=True)
