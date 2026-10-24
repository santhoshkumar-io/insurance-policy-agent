# Decision log

One entry per decision that someone could reasonably have made differently. Each records
what would reverse it, so the log stays useful rather than becoming a list of excuses.

---

## 2026-10-24 — Week 1 is a vertical slice, not an ingestion layer

**Decision.** Week 1 ends with upload → ask → cited answer working end-to-end, rather
than the brief's week 1 of "repo, API, DB, PDF extraction, chunk metadata."

**Why.** An ingestion layer with no consumer cannot be evaluated. Chunking decisions
only become visible once something retrieves from them, and the first time you see a
retrieved chunk is the first time you learn whether your chunking is any good.

**Reverses if.** Nothing foreseeable.

---

## 2026-10-24 — `tenant_id` on every table from migration 0001

**Decision.** The isolation boundary exists from the first migration. Authentication in
front of it is stubbed to a fixed dev tenant until week 5; the boundary itself is not.

**Why.** Isolation is a schema and query property. Retrofitting it means auditing every
query in the codebase and hoping none was missed. The cost now is a column and a helper.

**Reverses if.** Nothing. This is close to free.

---

## 2026-10-24 — Exact vector search, no ANN index

**Decision.** Sequential scan with cosine distance. No HNSW, no IVFFlat.

**Why.** Every query is filtered by tenant and policy set, which is exactly the case
where a pre-filtered ANN traversal loses recall. At MVP corpus size exact search is fast
and correct by construction.

**Reverses if.** Measured p50 retrieval latency exceeds ~500ms. At that point add HNSW
and document the filter selectivity and the measured recall loss.

---

## 2026-10-24 — In-process `BackgroundTasks`, no Celery

**Decision.** Ingestion runs in the API process. The `documents.status` column carries
the progress a queue would otherwise report.

**Why.** Celery adds a broker, a worker process, a serialisation boundary and a
deployment unit, to solve a problem that does not exist at single-developer volume.

**Reverses if.** Ingestion measurably blocks API responses, or ingestion needs to
survive an API restart.

---

## 2026-10-24 — Structured output plus post-validation for citations

**Decision.** The model returns JSON with explicit `citation_ids`, drawn only from ids
served to it. Every id is checked against the served set. One retry naming the
violation, then abstention.

**Why.** "Citations must point to retrieved source chunks" is only a guarantee if
something checks. Page numbers and sections are rendered from the database record, so
even a correct-looking fabricated page number cannot reach the user.

**Note.** This guarantees citation *existence*, not *relevance*. Whether the model cited
the right real clause is an evaluation question.

**Reverses if.** Nothing.

---

## 2026-10-24 — CLI as the week-1 interface

**Decision.** `policy ingest` / `policy ask` / `policy inspect`. Next.js in week 2.

**Why.** A browser cannot tell you whether retrieval is working, and building one slows
the loop at exactly the point where iteration speed matters most. `policy inspect` —
reading chunks by eye — caught more chunking problems than any test did.

**Reverses if.** Nothing; the CLI stays useful after the UI exists.

---

## 2026-10-24 — OCR deferred, but failure is loud

**Decision.** No OCR in week 1. A document with no extractable text on any page fails
ingestion with an explicit message rather than indexing blank pages.

**Why.** OCR on every page is slow and expensive, and most policy wordings are digital.
What is unacceptable is a scanned PDF silently producing an empty, queryable index.

**Reverses if.** Scanned policies turn out to be a primary input.

---

## 2026-10-24 — Table cell text is rebuilt from the word list, not `Table.extract()`

**Decision.** `rag/ingestion/pdf.py` reconstructs each table cell from the words whose
centre falls inside the cell bbox, falling back to `Table.extract()` only if that yields
nothing.

**Why.** Found while running the chunker against a real PDF: `Table.extract()` returned
`AIEngineering/DocumentIntelligence/SaaS` — every space gone. PDFs that position glyphs
individually carry no space characters, and the content stream has nothing to split on.
The word list splits on spatial gaps too, which recovers the boundaries.

This matters more here than it would elsewhere: table cells hold the sub-limits and
waiting periods, and `Roomrent1%ofsuminsuredperday` is unretrievable by keyword search
and close to meaningless as a citation.

**Reverses if.** Nothing; the fallback covers the case where bbox reconstruction fails.

---

## 2026-10-24 — Headings must not look like sentences

**Decision.** A line is rejected as a heading if it ends with a period and runs past six
words, if it carries a colon followed by more than 30 characters, or if it exceeds 14
words — regardless of how prominent its font is.

**Why.** Also found on a real PDF. Bulleted lists with a bold lead-in phrase
("**Policy upload:** Upload PDF policies and store document metadata.") were being read
as headings on font metrics alone, producing section paths with a whole sentence inside
them and fragmenting the surrounding prose into runt chunks. Font prominence identifies
a *candidate*; sentence shape rules it out.

**Reverses if.** A real policy wording turns out to use long sentence-shaped headings.
Worth re-checking once there are several insurers' documents in the corpus.
