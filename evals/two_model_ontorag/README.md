# Evaluation harness: needle → OntoRAG MCP → MobileMoE

See [REPORT.md](REPORT.md) for the write-up. Files:

| File | Purpose |
| --- | --- |
| `Dockerfile` | Eval image: ragas 0.4, trulens 2.14, sentence-transformers, langchain 0.3 pins |
| `data/rag_testset.jsonl` | 80 questions with references and gold chunk ids (amol dataset) |
| `data/mintaka_sample.jsonl` | 60 Mintaka questions, English + Italian, with gold labels |
| `collect.py` | Runs a test set through an aiproxy assistant, records contexts + answers |
| `mintaka_run.py` | Closed-book Mintaka against an openai-MobileMoE server |
| `score_basic.py` | Judge-free metrics |
| `score_ragas.py`, `score_trulens.py`, `score_ares.py` | LLM-judged metrics (need `JUDGE_*`) |
| `runs/` | Collected records from the reported runs |
| `out/` | Judged scorer outputs (gpt-4.1-mini via OpenRouter) |
| `report_judged.py` | Turns `out/*.json` into the report tables |

The Mintaka test split itself (`mintaka_test.json`, 9.6 MB) is fetched from
https://github.com/amazon-science/mintaka and not committed.
