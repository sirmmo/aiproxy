"""Judge-free metrics over collected records: retrieval hit rate on gold chunks,
answer mentions the entity, negative rejection on unanswerable questions, timing."""
import json, re, sys, unicodedata
from collections import defaultdict

REJECT = re.compile(r"(no information|nothing on|does not contain|doesn't contain|not contain|no data|not found|not aware|unable to (find|provide)|non (ho|contiene|risult)|nessun[ao]? (informazion|risultat)|cannot find|no relevant|not mention)", re.I)
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode().lower())

files = sys.argv[1:]
for path in files:
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(list)
    for r in rows: by[(r["kind"], r["lang"])].append(r)
    print(f"\n=== {path.split('/')[-1]}  ({len(rows)} records, assistant={rows[0]['assistant'] if rows else '?'})")
    print(f"{'kind/lang':22} {'n':>3} {'errors':>6} {'ret-hit':>8} {'ent-in-ctx':>10} {'ent-in-ans':>10} {'rejects':>8} {'len-cap':>8} {'mean s':>7} {'tok':>5}")
    for key, rs in sorted(by.items()):
        n = len(rs); errs = sum(1 for r in rs if r["error"])
        hit = sum(1 for r in rs if set(r["retrieved_ids"]) & set(r["gold_ids"])) / max(1, sum(1 for r in rs if r["gold_ids"]))
        ent = sum(1 for r in rs if r["entity"] and norm(r["entity"]) in norm(r["answer"])) / max(1, sum(1 for r in rs if r["entity"]))
        ectx = sum(1 for r in rs if r["entity"] and norm(r["entity"]) in norm(" ".join(r["contexts"]))) / max(1, sum(1 for r in rs if r["entity"]))
        rej = sum(1 for r in rs if REJECT.search(r["answer"] or "")) / n
        cap = sum(1 for r in rs if r["finish"] == "length") / n
        secs = sum(r["seconds"] for r in rs) / n; tok = sum(r["completion_tokens"] for r in rs) / n
        print(f"{key[0]+'/'+key[1]:22} {n:3d} {errs:6d} {hit:8.2f} {ectx:10.2f} {ent:10.2f} {rej:8.2f} {cap:8.2f} {secs:7.1f} {tok:5.0f}")
