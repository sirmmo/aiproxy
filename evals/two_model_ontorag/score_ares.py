"""ARES-style scoring (Saad-Falcon et al., 2024). The ares-ai package did not install
in this environment, so this reproduces its UES/IDP judge protocol: three binary
judgements per record, context relevance, answer faithfulness and answer relevance,
made by an LLM judge with the paper's yes/no framing, then averaged. ARES's
prediction-powered inference step needs a human-labelled set and is not applied."""
import json, os, sys, statistics
import judge
judge.require()

src, out = sys.argv[1], sys.argv[2]
limit = int(os.environ.get("LIMIT", "0") or 0)
rows = judge.load_records(src)
if limit: rows = rows[:limit]
cl = judge.client()

CR = ("Given the following question and document, you must analyze the provided document and determine whether it is "
      "sufficient for answering the question. In your evaluation, you should consider the content of the document and how "
      "it relates to the provided question. Output your final verdict by strictly following this format: \"[[Yes]]\" if the "
      "document is sufficient and \"[[No]]\" if the document provided is not sufficient. Do not provide any additional explanation.\n\n"
      "Question: {q}\n\nDocument: {d}")
AF = ("Given the following question, document, and answer, you must analyze the provided answer and determine whether it is "
      "faithful to the contents of the document. The answer must not offer new information beyond the context provided in the "
      "document. The answer also must not contradict information provided in the document. Output your final verdict by strictly "
      "following this format: \"[[Yes]]\" if the answer is faithful to the document and \"[[No]]\" if the answer is not faithful "
      "to the document. Do not provide any additional explanation.\n\nQuestion: {q}\n\nDocument: {d}\n\nAnswer: {a}")
AR = ("Given the following question, document, and answer, you must analyze the provided answer and document before determining "
      "whether the answer is relevant for the provided question. In your evaluation, you should consider whether the answer "
      "addresses all aspects of the question and provides only correct information from the document for answering the question. "
      "Output your final verdict by strictly following this format: \"[[Yes]]\" if the answer is relevant for the given question "
      "and \"[[No]]\" if the answer is not relevant for the given question. Do not provide any additional explanation.\n\n"
      "Question: {q}\n\nDocument: {d}\n\nAnswer: {a}")

def yes(text: str) -> int:
    t = text.lower()
    return 1 if "[[yes]]" in t or (t.strip().startswith("yes") and "[[no]]" not in t) else 0

recs = []
for r in rows:
    doc = "\n\n".join(r["contexts"][:6]) if r["contexts"] else "(no document retrieved)"
    try:
        cr = yes(judge.ask(cl, CR.format(q=r["question"], d=doc), 10))
        af = yes(judge.ask(cl, AF.format(q=r["question"], d=doc, a=r["answer"] or ""), 10))
        ar = yes(judge.ask(cl, AR.format(q=r["question"], d=doc, a=r["answer"] or ""), 10))
        recs.append({"qid": r["qid"], "kind": r["kind"], "lang": r["lang"], "assistant": r["assistant"], "answerable": r["answerable"],
                     "context_relevance": cr, "answer_faithfulness": af, "answer_relevance": ar})
        print(f"[ares] {r['qid']} cr={cr} af={af} ar={ar}", flush=True)
    except Exception as e:
        recs.append({"qid": r["qid"], "kind": r["kind"], "lang": r["lang"], "assistant": r["assistant"], "error": str(e)[:200]})
ok = [x for x in recs if "error" not in x]
def agg(sub): return {m: round(statistics.mean(x[m] for x in sub), 3) for m in ("context_relevance", "answer_faithfulness", "answer_relevance")} | {"n": len(sub)}
by_kind = {"/".join(k): agg([x for x in ok if (x["kind"], x["lang"]) == k]) for k in sorted({(x["kind"], x["lang"]) for x in ok})}
json.dump({"source": src, "judge": judge.MODEL, "protocol": "ARES UES/IDP judge prompts, no PPI", "n": len(ok), "errors": len(recs) - len(ok),
           "aggregate": agg(ok) if ok else {}, "by_kind": by_kind, "records": recs}, open(out, "w"), indent=1, ensure_ascii=False)
print("aggregate:", agg(ok) if ok else {}); print("wrote", out)
