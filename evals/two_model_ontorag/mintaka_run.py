"""Closed-book Mintaka QA straight against an openai-MobileMoE server, EN and IT.
Scores 'label contained in answer' (normalized), the usual lenient match for Mintaka."""
import json, os, re, sys, time, unicodedata
from openai import OpenAI

base, model_tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
c = OpenAI(base_url=base, api_key="x", timeout=600)
rows = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "data", "mintaka_sample.jsonl"))]
done = {(json.loads(l)["id"], json.loads(l)["lang"]) for l in open(out)} if os.path.exists(out) else set()

def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()

SYS = {"en": "Answer the question with just the answer, in a few words.",
       "it": "Rispondi alla domanda con la sola risposta, in poche parole."}
with open(out, "a") as f:
    for r in rows:
        for lang in ("en", "it"):
            if (r["id"], lang) in done:
                continue
            q = r[f"question_{lang}"]; labels = r["labels_en"] + r["labels_it"]
            t = time.time()
            resp = c.chat.completions.create(model=model_tag, messages=[{"role": "system", "content": SYS[lang]}, {"role": "user", "content": q}], max_tokens=48, temperature=0)
            ans = resp.choices[0].message.content or ""
            hit = any(norm(l) and norm(l) in norm(ans) for l in labels)
            rec = {"id": r["id"], "lang": lang, "complexity": r["complexity"], "question": q, "labels": labels, "answer": ans, "hit": hit, "seconds": round(time.time() - t, 1)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"[{model_tag}/{lang}] hit={hit} {rec['seconds']}s :: {q[:60]!r} -> {ans[:60]!r}", flush=True)
