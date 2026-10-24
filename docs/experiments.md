# Retrieval experiments

Empty by design. Week 3 fills it.

The rule, so that this file is worth reading: every row is produced by the harness
against the frozen question set in `evals/`, and **negative results stay in the table**.
"Reranking cost 180ms and gained 2 points of Recall@5 — not adopted" is a more useful
row than a silent omission, and it is the row that shows the decision was made on
evidence.

| Experiment | Recall@5 | MRR | Citation acc. | p50 latency | Cost / 100q | Adopted | Notes |
|---|---|---|---|---|---|---|---|
| Vector only (baseline) | — | — | — | — | — | — | week 3 |
| Keyword (`ts_rank_cd`) | — | — | — | — | — | — | week 3 |
| Hybrid (RRF, k=60) | — | — | — | — | — | — | week 3 |
| Hybrid + reranker | — | — | — | — | — | — | week 3 |
| + query rewriting | — | — | — | — | — | — | week 3 |

Note on naming: PostgreSQL full-text ranking is `ts_rank_cd`, **not BM25**. The brief's
stack table says BM25; stock PostgreSQL does not provide it. This project reports what
it implements. A real BM25 comparison (via `rank_bm25` in Python over this corpus size)
runs as a separate row.
