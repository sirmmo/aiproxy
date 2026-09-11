"""RAGAS metrics over collected records. Faithfulness and answer relevancy for every
answerable question; context precision/recall and answer correctness where a
reference exists (entity summaries). Embeddings are local (MiniLM); the judge is
the JUDGE_* endpoint."""
import json, os, sys
import judge
judge.require()
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness

src, out = sys.argv[1], sys.argv[2]
limit = int(os.environ.get("LIMIT", "0") or 0)
rows = [r for r in judge.load_records(src) if r["answerable"] and r["contexts"]]
if limit: rows = rows[:limit]
llm = LangchainLLMWrapper(ChatOpenAI(model=judge.MODEL, base_url=judge.BASE, api_key=judge.KEY or "x", temperature=0, timeout=180, max_retries=3))
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

def run(subset, metrics, tag):
    if not subset: return {}
    ds = EvaluationDataset.from_list([{"user_input": r["question"], "retrieved_contexts": r["contexts"][:4], "response": r["answer"] or "",
                                       **({"reference": r["reference"]} if r["reference"] else {})} for r in subset])
    res = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb, raise_exceptions=False, show_progress=False,
                   run_config=RunConfig(max_workers=4, timeout=240, max_retries=6, max_wait=60))
    df = res.to_pandas()
    per = df.to_dict(orient="records")
    for r, p in zip(subset, per):
        p["qid"], p["kind"], p["lang"], p["assistant"] = r["qid"], r["kind"], r["lang"], r["assistant"]
    agg = {m: round(float(df[m].mean()), 3) for m in df.columns if m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness")}
    print(f"[ragas:{tag}] n={len(subset)} {agg}", flush=True)
    return {"n": len(subset), "aggregate": agg, "records": per}

with_ref = [r for r in rows if r["reference"]]
no_ref = [r for r in rows if not r["reference"]]
result = {"source": src, "judge": judge.MODEL,
          "with_reference": run(with_ref, [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness], "with_reference"),
          "no_reference": run(no_ref, [faithfulness, answer_relevancy], "no_reference")}
json.dump(result, open(out, "w"), indent=1, ensure_ascii=False, default=str)
print("wrote", out)
