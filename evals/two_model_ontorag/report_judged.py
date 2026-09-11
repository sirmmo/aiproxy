"""Assemble markdown tables from the judged scorer outputs in out/."""
import glob, json, os, statistics, sys
from collections import defaultdict

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
NAMES = {"ontorag-chat": "ontorag-chat (M)", "ontorag-chat-s": "ontorag-chat-s (S)", "ontorag-chat-wide": "ontorag-chat (M, 14k cap, 10 q)", "mobilemoe-only": "closed-book control (S)"}
ORDER = ["ontorag-chat", "ontorag-chat-s", "ontorag-chat-wide", "mobilemoe-only"]

def load(prefix):
    res = {}
    for f in glob.glob(f"{OUT}/{prefix}-*.json"):
        name = os.path.basename(f)[len(prefix) + 1:-5]
        res[name] = json.load(open(f))
    return res

def mean(xs): return round(statistics.mean(xs), 2) if xs else None
def fmt(v): return "–" if v is None else f"{v:.2f}"

# ---- ARES-style
ares = load("score_ares")
print("#### ARES-style judge (yes/no rates)\n")
print("| Assistant | Kind / lang | n | context relevance | answer faithfulness | answer relevance |")
print("| --- | --- | ---: | ---: | ---: | ---: |")
for a in ORDER:
    if a not in ares: continue
    for k, v in ares[a]["by_kind"].items():
        print(f"| {NAMES[a]} | {k} | {v['n']} | {fmt(v['context_relevance'])} | {fmt(v['answer_faithfulness'])} | {fmt(v['answer_relevance'])} |")
print()

# ---- TruLens
tl = load("score_trulens")
print("#### TruLens RAG triad (0–1)\n")
print("| Assistant | Kind / lang | n | context relevance | groundedness | answer relevance |")
print("| --- | --- | ---: | ---: | ---: | ---: |")
for a in ORDER:
    if a not in tl: continue
    for k, v in tl[a]["by_kind"].items():
        print(f"| {NAMES[a]} | {k} | {v['n']} | {fmt(v['context_relevance'])} | {fmt(v['groundedness'])} | {fmt(v['answer_relevance'])} |")
    if tl[a].get("errors"): print(f"| {NAMES[a]} | errors | {tl[a]['errors']} | | | |")
print()

# ---- RAGAS
rg = load("score_ragas")
print("#### RAGAS (0–1)\n")
print("| Assistant | Kind / lang | n | faithfulness | answer relevancy | context precision | context recall | answer correctness |")
print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for a in ORDER:
    if a not in rg: continue
    recs = []
    for part in ("with_reference", "no_reference"):
        recs += (rg[a].get(part) or {}).get("records") or []
    by = defaultdict(list)
    for r in recs: by[(r["kind"], r["lang"])].append(r)
    for (kind, lang), rs in sorted(by.items()):
        cols = []
        for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"):
            vals = [r[m] for r in rs if isinstance(r.get(m), (int, float)) and r[m] == r[m]]
            cols.append(fmt(mean(vals)) if vals else "–")
        print(f"| {NAMES[a]} | {kind} / {lang} | {len(rs)} | " + " | ".join(cols) + " |")
print()
