# Evaluating the needle → OntoRAG → MobileMoE stack

RAG evaluation of the two-model assistant served by aiproxy: needle-openai (45M
parameters) decides which OntoRAG MCP tool to call, the tool runs against the
`amol-ontorag` Ars Magica knowledge graph, and openai-MobileMoE writes the answer
from the retrieved text. Everything below ran on one CPU-only host (two Xeon
E5-2640 v4, 2016, AVX2 only) with every component in Docker.

## Configurations under test

| Assistant | Decides tools | Answers | Retrieval |
| --- | --- | --- | --- |
| `ontorag-chat` | needle-2 | MobileMoE-M-QAT (528M active) | ontorag-mcp, `ontology` mode, `tool_fallback` on |
| `ontorag-chat-s` | needle-2 | MobileMoE-S-QAT (272M active) | same |
| `mobilemoe-only` | (tools offered to MobileMoE-S, which ignores them) | MobileMoE-S-QAT | none: closed-book control |

Common settings: one decision per turn (`tool_max_rounds: 1`), three tools
exposed (`answer`, `search_entities`, `entity_chunks`) with rewritten one-line
descriptions, pinned arguments (`k: 3`, `expand: 1`, `limit: 5`), tool results
clipped to 5000 characters, `temperature: 0`, answer cap 1000 tokens,
repetition penalty 1.15 on the MobileMoE servers. The exact config is
`examples/two_model_ontorag.yaml`.

## Test sets

**RAG set, 80 questions** (`data/rag_testset.jsonl`), derived without any LLM:

- 40 `entity_named` English questions, "Tell me about X.", for entities of the
  amol graph that have a summary. Reference answer = the entity's summary from
  `ontology/entities.jsonl`; gold chunks = the chunk ids linked to the entity,
  taken from `ontorag-mcp/eval/queries.jsonl` (v0.4.2 provenance layer).
- 20 of the same kind in Italian, "Parlami di X.", with the English reference.
- 10 `paraphrase` questions: the entity's summary with its name masked, gold
  chunks only (no reference), to test retrieval when nothing is named.
- 10 `unanswerable` questions about things absent from the corpus (Gandalf,
  the capital of France, ...), to measure negative rejection.

**Mintaka, 60 questions** (`data/mintaka_sample.jsonl`): entity-answer
questions from the Mintaka test split (Sen et al., 2022) in English and their
Italian translations, generic, intersection and multihop types, with gold
labels in both languages. Mintaka asks about Wikidata, not Ars Magica, so it is
run closed-book straight against the MobileMoE servers: it measures the answer
model's general knowledge and its Italian degradation, not the RAG stack.

## Metrics

Judge-free (`score_basic.py`):

- **ret-hit**: a gold chunk id appears in the retrieved text. Only `answer` and
  `entity_chunks` return chunk ids, so this is a lower bound on retrieval.
- **ent-in-ctx**: the asked entity's label appears in the retrieved text
  (covers `search_entities`, which returns entity cards without chunk ids).
- **ent-in-ans**: the entity's label appears in the answer.
- **rejects**: the answer says the graph has nothing (pattern match), reported
  for every kind; it should be high on `unanswerable` and low elsewhere.
- **len-cap**: answers that hit the token cap; mean seconds and tokens.

LLM-judged, all on the same records and the same judge endpoint (`JUDGE_*`):

- **RAGAS** (`score_ragas.py`, ragas 0.4.3): faithfulness, answer relevancy
  (local MiniLM embeddings), and, where a reference exists, context precision,
  context recall and answer correctness.
- **TruLens RAG triad** (`score_trulens.py`, trulens 2.14): context relevance,
  groundedness, answer relevance, using TruLens's own feedback functions.
- **ARES-style** (`score_ares.py`): the `ares-ai` package does not install on
  Python 3.12 (a legacy dependency fails metadata generation), so the script
  reproduces ARES's UES/IDP judge protocol: three yes/no judgements per record,
  context relevance, answer faithfulness, answer relevance, averaged. ARES's
  prediction-powered-inference correction needs a human-labelled set and was
  not applied.

Mintaka (`mintaka_run.py`): lenient match, a gold label (English or Italian)
contained in the normalised answer.

## Results

### Judge-free metrics (RAG set)

`ret-hit` = a gold chunk id among the retrieved passages; `ent-in-ctx` = the
asked entity appears in the retrieved text; `ent-in-ans` = it appears in the
answer; `rejects` = the answer says the graph has nothing; `cap` = answers that
hit the 1000-token limit.

| Assistant | Kind / lang | n | ret-hit | ent-in-ctx | ent-in-ans | rejects | cap | mean s | tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ontorag-chat (M) | entity_named / en | 40 | 0.05 | 0.95 | 0.95 | 0.05 | 0.00 | 72 | 185 |
| ontorag-chat (M) | entity_named / it | 20 | 0.00 | 0.80 | 0.90 | 0.10 | 0.00 | 28 | 194 |
| ontorag-chat (M) | paraphrase / en | 10 | 0.00 | 0.50 | 0.20 | 0.30 | 0.20 | 65 | 381 |
| ontorag-chat (M) | unanswerable / en | 10 | – | – | – | **0.50** | 0.00 | 19 | 99 |
| ontorag-chat-s (S) | entity_named / en | 40 | 0.05 | 0.95 | 0.68 | 0.20 | 0.17 | 48 | 299 |
| ontorag-chat-s (S) | paraphrase / en | 10 | 0.00 | 0.50 | 0.20 | 0.00 | 0.30 | 72 | 609 |
| ontorag-chat-s (S) | unanswerable / en | 10 | – | – | – | 0.30 | 0.20 | 32 | 310 |
| mobilemoe-only (S, no retrieval) | entity_named / en | 40 | – | – | 0.88 | 0.03 | 0.00 | 18 | 190 |
| mobilemoe-only (S, no retrieval) | paraphrase / en | 10 | – | – | 0.10 | 0.00 | 0.00 | 19 | 206 |
| mobilemoe-only (S, no retrieval) | unanswerable / en | 10 | – | – | – | 0.00 | 0.00 | 6 | 66 |

Tool choice by needle over the 80 M-assistant turns: `answer` 57 times,
`search_entities` 28, `entity_chunks` never; the fallback fired on 10 turns
(needle declined), 4 turns carried two calls. The S assistant saw the same
decisions, as expected, since needle decides.

### The clipping finding

`ret-hit` is near zero not because retrieval misses but because the passages
never reach the model. The `answer` tool returns `query`, `matched_entities`,
`ontology_facts` and only then `passages` (each with a `cite` chunk id), and
the 5000-character `tool_result_max_chars` clip falls inside `ontology_facts`
for most results: only 7 of the 70 answerable M records and 6 of 50 S records
contained a single chunk id. What the answer model actually reads is the entity
cards (`ent-in-ctx` 0.95), which is why the answers are grounded summaries of
entity descriptions rather than citations of source text, and why "mention the
chunk ids you relied on" in the system prompt is never satisfied.

Two fixes, either sufficient: raise `tool_result_max_chars` (a 10-question
control with 14000 is reported in the addendum), or have `ontorag-mcp`'s
`answer` put `passages` first and keep `ontology_facts` short.

### What the judge-free numbers say

- **Retrieval brings back the right entity** 95% of the time in English and
  80% in Italian; needle copies the entity name into `search_entities` or
  `answer` reliably, even from Italian questions.
- **M beats S on answer discipline**: S echoes or rejects on 20% of named
  questions it had context for, hits the token cap on 17%, and writes 60%
  longer answers; M rejects 5% and never hits the cap.
- **Negative rejection is the weak spot.** On questions the corpus cannot
  answer, M says so only half the time and S 30%; the closed-book control never
  does. With the fallback forcing a retrieval on every turn, unrelated passages
  come back and the model sometimes answers from them or from its own
  knowledge. The closed-book control answers everything, fast and unfounded.
- **Paraphrase questions** (entity name masked) are where retrieval genuinely
  struggles: the entity appears in the context half the time and in the answer
  20%, and M declines on 30%.
- **Italian in, English out**: 90% of Italian questions got an answer naming the
  entity, but in English, under an English system prompt.

### Mintaka, closed-book (general knowledge, not the RAG stack)

| Model | Lang | n | hit | generic | intersection | multihop | mean s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileMoE-M-QAT | en | 60 | **0.50** | 0.52 | 0.57 | 0.30 | 1.8 |
| MobileMoE-M-QAT | it | 60 | 0.33 | 0.31 | 0.43 | 0.20 | 2.3 |
| MobileMoE-S-QAT | en | 60 | 0.38 | 0.45 | 0.43 | 0.10 | 1.7 |
| MobileMoE-S-QAT | it | 60 | 0.22 | 0.14 | 0.38 | 0.10 | 2.0 |

Lenient containment match on a 60-question sample; Mintaka's published
baselines are not comparable (different sample and metric). The pattern is
clear enough: M knows more than S, both lose about 16 points in Italian, and
multi-hop questions are largely out of reach for either.

### LLM-judged metrics (RAGAS, TruLens, ARES-style)

_Pending: the scorers are ready and smoke-tested but need a judge model key
(`JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL`). No key was available on the
evaluation host._

### Addendum: 14000-character cap control

The same first ten English `entity_named` questions, re-run through a copy of
`ontorag-chat` with `tool_result_max_chars: 14000` (added at runtime through
the admin API, everything else identical):

| | 5000-char cap | 14000-char cap |
| --- | ---: | ---: |
| Records whose retrieved text contains any chunk id | 2 / 10 | 5 / 10 |
| Gold chunk retrieved | 2 / 10 | 5 / 10 |
| Entity in retrieved text | 10 / 10 | 10 / 10 |
| Chunk ids cited in the answer | 0 | 0 |
| Mean answer length (tokens) | 218 | 153 |

With the wider cap five of the seven `answer` calls delivered their passages
and every one of those retrieved a gold chunk, against two before, which
confirms the clip as the cause. `search_entities` results carry no chunk ids
in either setting, so the ceiling for this metric is the share of turns needle
routes to `answer` (about two thirds). The answers were also shorter and read
as direct summaries. Even with the passages present the model never quoted a
chunk id, so "mention the chunk ids" is a request this model size does not
honour; citations should be attached by the gateway from the `cite` fields,
not by the model. Timings are not comparable: the 5000-cap run of these
questions overlapped with an image build on the same CPU.

## Caveats

- N is small (40/20/10/10 and 60) because M answers take about a minute each
  on this CPU; treat differences under ten points as noise.
- Retrieval hit rates depend on which tool needle chose; `search_entities`
  results carry no chunk ids, which is why `ent-in-ctx` is reported alongside.
- The judge model and its prompts are a source of bias shared by all three
  judged frameworks; the same judge is used everywhere so comparisons across
  assistants are fair even if absolute values are not.
- Italian answers are judged against English references.

## Reproducing

```bash
docker build -t rag-eval:dev .                      # ragas, trulens, sentence-transformers
python collect.py ontorag-chat data/rag_testset.jsonl runs/ontorag-chat.jsonl
python mintaka_run.py http://127.0.0.1:8001/v1 MobileMoE-M-QAT runs/mintaka-m.jsonl
python score_basic.py runs/*.jsonl
JUDGE_BASE_URL=https://openrouter.ai/api/v1 JUDGE_API_KEY=... JUDGE_MODEL=... \
  docker run --rm --network host -v $PWD:/work -v ~/.cache/huggingface:/cache/huggingface \
  -e JUDGE_BASE_URL -e JUDGE_API_KEY -e JUDGE_MODEL rag-eval:dev python /work/score_ragas.py runs/ontorag-chat.jsonl out/ragas-m.json
```
