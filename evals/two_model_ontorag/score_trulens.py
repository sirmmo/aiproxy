"""TruLens RAG triad (context relevance, groundedness, answer relevance) using
TruLens's own feedback implementations on the JUDGE_* endpoint, computed per
record without the session database."""
import json, os, sys, statistics
import judge
judge.require()
from trulens.providers.openai import OpenAI as TLOpenAI

src, out = sys.argv[1], sys.argv[2]
limit = int(os.environ.get("LIMIT", "0") or 0)
rows = [r for r in judge.load_records(src) if r["contexts"]]
if limit: rows = rows[:limit]
provider = TLOpenAI(model_engine=judge.MODEL, base_url=judge.BASE, api_key=judge.KEY or "x")
# TruLens 2.x probes the Responses API and structured outputs first; most
# OpenAI-compatible judges (OpenRouter, local servers) only speak chat completions.
provider._set_capabilities({"responses_api": False, "cfg": False, "structured_outputs": False, "json_mode": False})
recs = []
for r in rows:
    ctxs = r["contexts"][:6]
    try:
        crel = statistics.mean(provider.context_relevance(r["question"], c) for c in ctxs)
        grounded, _ = provider.groundedness_measure_with_cot_reasons("\n\n".join(ctxs), r["answer"] or "")
        arel = provider.relevance(r["question"], r["answer"] or "")
        recs.append({"qid": r["qid"], "kind": r["kind"], "lang": r["lang"], "assistant": r["assistant"],
                     "context_relevance": round(float(crel), 3), "groundedness": round(float(grounded), 3), "answer_relevance": round(float(arel), 3)})
        print(f"[trulens] {r['qid']} ctx_rel={crel:.2f} grounded={grounded:.2f} ans_rel={arel:.2f}", flush=True)
    except Exception as e:
        recs.append({"qid": r["qid"], "kind": r["kind"], "lang": r["lang"], "assistant": r["assistant"], "error": str(e)[:200]})
ok = [x for x in recs if "error" not in x]
agg = {k: round(statistics.mean(x[k] for x in ok), 3) for k in ("context_relevance", "groundedness", "answer_relevance")} if ok else {}
by_kind = {}
for k in sorted({(x["kind"], x["lang"]) for x in ok}):
    sub = [x for x in ok if (x["kind"], x["lang"]) == k]
    by_kind["/".join(k)] = {m: round(statistics.mean(x[m] for x in sub), 3) for m in ("context_relevance", "groundedness", "answer_relevance")} | {"n": len(sub)}
json.dump({"source": src, "judge": judge.MODEL, "n": len(ok), "errors": len(recs) - len(ok), "aggregate": agg, "by_kind": by_kind, "records": recs}, open(out, "w"), indent=1, ensure_ascii=False)
print("aggregate:", agg); print("wrote", out)
