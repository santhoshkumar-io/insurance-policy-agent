# Insurance Policy Intelligence AI

A RAG platform that answers questions about insurance policy documents with citations
that are **verified against the retrieved text**, and that abstains rather than guessing.

Built from `insurance_policy_intelligence_ai_project_brief.pdf`. The engineering
analysis of that brief — what it gets right, where it was under-specified, and the
decisions taken in response — is in [plan.md](plan.md).

> **Status: phase 3 of 6 complete.** The benchmark now runs on **real IRDAI-filed LIC
> policy wordings**, which overturned two conclusions from the previous phase and exposed
> six chunker defects that every test had passed. See
> [docs/experiments.md](docs/experiments.md). The web UI, policy comparison, real auth,
> tracing and deployment are still ahead. Every number below came out of
> `policy experiments`.

---

## What is actually built

| Capability | State |
|---|---|
| PDF ingestion with page-accurate metadata | ✅ |
| Structure-aware chunking (headings, clause numbering, tables, fallback) | ✅ |
| Running header/footer removal | ✅ |
| OpenAI embeddings, model recorded per vector | ✅ |
| Exact tenant-scoped vector retrieval | ✅ |
| Grounded generation with structured output | ✅ |
| **Citation validation** — invented ids are rejected, not rendered | ✅ |
| Abstention with explicit triggers | ✅ |
| Tenant isolation in the schema and every query | ✅ (auth stubbed) |
| REST API + CLI | ✅ |
| Keyword retrieval (`ts_rank_cd`) + RRF hybrid | ✅ built; beaten by BM25 on a real corpus |
| Real BM25 arm (`rank_bm25`) | ✅ built; **best recall**, but not deployable as built |
| Cross-encoder reranking | ✅ built, off by default; best MRR, no recall gain |
| Section-path retrieval arm | ✅ built, **rejected on measurement** — see below |
| Evaluation harness: Recall@k, MRR, nDCG, abstention, conditional coverage, cost | ✅ |
| Tri-state answerability (yes / no / conditional) | ✅ |
| Two separate benchmark suites (real + synthetic regression) | ✅ |
| 47-question benchmark on real LIC wordings | ✅ |
| Re-index without re-upload | ✅ |
| Candidate question drafting for the benchmark | ✅ |
| Structure-aware chunking of real IRDAI-filed wordings | ✅ |
| BM25 inside Postgres (ParadeDB `pg_search`) | justified by measurement; phase 4 |
| OCR for scanned pages | not needed — both real documents extract cleanly |
| Next.js dashboard | phase 3 |
| Policy comparison | phase 4 |
| Real auth, tracing | phase 5 |
| Docker deploy, CI | phase 6 |

---

## Quick start

```bash
cp .env.example .env            # add your OPENAI_API_KEY
docker compose up -d            # PostgreSQL 17 + pgvector on port 5433

uv venv --python 3.12
uv pip install -e ".[dev]"
source .venv/bin/activate

alembic upgrade head
pytest -q -m "not db"           # unit tests, no database or network needed
pytest -q                       # everything, including tenant isolation
```

Then ingest a policy and ask it something:

```bash
policy ingest samples/health-policy.pdf --product "Health Guard" --insurer "Acme"
policy ask "What is the waiting period for pre-existing diseases?"
policy ask "What is the premium for a family of four?"   # expect an abstention
policy inspect <document_id> --page 14                   # read chunks by eye
```

Or run the API:

```bash
uvicorn services.api.main:app --reload    # docs at http://127.0.0.1:8000/docs
```

---

## What the benchmark says

47 questions (41 answerable, 3 conditional, 6 unanswerable) hand-verified against two real
LIC policy wordings — *Digi Term* and *Jeevan Raksha*, 17 pages and 49 chunks each. Full
table, method and caveats in [docs/experiments.md](docs/experiments.md).

| Configuration | Recall@5 | MRR |
|---|---|---|
| vector | 0.949 | 0.911 |
| keyword (`ts_rank_cd`) | 0.872 | 0.608 |
| bm25 (`rank_bm25`) | 0.974 | 0.795 |
| hybrid RRF(vector + `ts_rank_cd`) | 0.897 | 0.782 |
| **hybrid RRF(vector + bm25)** | **1.000** | 0.898 |
| hybrid RRF(vector + `ts_rank_cd` + section) | 0.923 | 0.772 |
| hybrid RRF(vector + bm25 + section) | 0.897 | 0.750 |
| **hybrid + cross-encoder** | 0.949 | **0.932** |

**Three results worth reading the detail for.**

**The previous phase's verdict was wrong, and the caveat that said so was the useful
part.** Phase 2 concluded plain vector beat every hybrid — on an 8-chunk synthetic
document, while stating that could not generalise. On a real corpus
`hybrid RRF(vector + bm25)` retrieves the evidence for **every** answerable question
(1.000) where vector alone does not (0.949). That conclusion is withdrawn.

**Real BM25 beats `ts_rank_cd` by 0.187 MRR** — against 0.032 on the synthetic corpus,
which was reported as noise at the time. The mechanism is IDF, and these documents are
dense with the rare tokens that rewards: `512N356V02`, `Section 45`, `Unexpired Risk
Premium Value`. Running BM25 as a measured experiment instead of asserting `ts_rank_cd`
was adequate is what made that visible — and it fires the reversal condition recorded in
[docs/decisions.md](docs/decisions.md), so ParadeDB `pg_search` is now justified.

**But the winning configuration cannot ship.** The in-memory BM25 arm re-indexes the whole
corpus on every query — a benchmark tool, as its own docstring says. So the default stays
`vector` until `pg_search` makes a deployable hybrid measurable. Flipping a setting on the
strength of a number produced by code that cannot serve traffic would be the wrong lesson
to take from a benchmark.

### Section-aware retrieval was my top recommendation, and it lost

Phase 2 ended by naming this "the first thing to try in the next phase". Built, measured,
rejected: adding the section arm to the best pair costs **10 points of recall and 15 of
MRR**, and the synthetic corpus agrees.

Why it fails is the useful part. A section path identifies a *region*, not a *passage* —
eight chunks share `PART – D: CONDITIONS RELATED TO SERVICING ASPECTS`, so a query
matching that path returns all eight at high rank and each contributes its reciprocal to
the fusion. RRF treats every arm as an equal voter and this signal is not of equal
quality. The diagnosis still stands; section metadata belongs as a tie-breaker or filter,
not a fourth vote. `rag/retrieval/section.py` stays behind a flag.

### The chunker that passed every test was broken on real documents

Section paths on these documents looked like `` >  > PART- A`` and
`18. Endorsement means conditions attached/ affixed to this Policy...`, while every
chunking test passed. The synthetic fixture had 16pt headings over 10pt body text; in the
LIC documents **564 body lines and 138 heading lines are all 10.0pt**, so prominence
cannot find headings at all. Heading detection is now anchored on the IRDAI-prescribed
`PART A..G` backbone.

The worst defect was caught by a *benchmark question*, not a test: "What is the UIN of
this plan?" was unanswerable because the UIN sits in the running header of all 17 pages
and furniture detection had deleted it from the corpus entirely. Furniture is now
deduplicated rather than deleted — kept once on the title page, stripped thereafter.

## How a question is answered

```
question
   ↓ embed
   ↓ retrieve      vector | keyword | bm25 | hybrid(RRF), filtered by tenant + policies
   ↓ rerank        optional cross-encoder, off by default (it earned nothing)
   ↓ confidence    cosine similarity attached to every result, whichever arm found it
   ↓ prompt        sources in a USER turn, inside <source id=…> blocks
   ↓ generate      structured JSON: answerable, answer, claims[].citation_ids[]
   ↓ validate      every id must be one that was served — else retry once, then abstain
   ↓ render        page and section read from the database, never from model output
answer + citations
```

### The three guarantees

**1. A citation cannot be fabricated.** The model is given a fixed set of ids and may
use nothing else. Every returned id is checked against the set actually served. A
violation triggers one retry naming the offending id; a second failure abstains. Page
numbers and section paths are then looked up from the chunk record, so even a plausible
fabricated page number has no path to the user.

This guarantees *existence*. Whether the model cited the *right* real clause is an
evaluation question, measured in week 5 — no validator can answer it.

**2. The system abstains on three explicit triggers**, not on vibes: nothing retrieved;
best retrieval score below `ABSTAIN_SCORE_THRESHOLD`; or the model reporting
insufficient evidence. Both error directions get measured, because a system that
abstains on everything scores perfectly on groundedness and is useless.

**3. Policy text is data, never instruction.** The system prompt contains no document
content, so nothing inside a PDF can rewrite the system's role. Retrieved text arrives
in a user turn inside `<source>` delimiters. A red-team fixture — a PDF carrying
"ignore all previous instructions" — is asserted in the test suite.

---

## Chunking

The hardest part of this project, and the part the brief compressed into one line.
Policy wordings are hostile to naive extraction: deep clause numbering where the section
path *is* the answer's address, and tables holding the sub-limits and waiting periods
that agents actually ask about — tables that `get_text()` flattens into unusable word
soup.

[`rag/chunking/chunker.py`](rag/chunking/chunker.py) therefore:

1. reads PyMuPDF **spans** with font metrics, not plain text, and sizes each line by its
   **dominant** span rather than its largest — a bold `18.` marker was otherwise
   promoting whole definition sentences to top-level headings;
2. anchors level 1 on the IRDAI-prescribed **`PART A..G` backbone**, because in real
   wordings headings share the body point size and prominence cannot find them;
3. detects numbered clauses beneath their PART, never at level 1, so the structural
   parent can never be displaced;
4. rejects body text that only looks like a heading — numeric furniture, definition
   sentences (`N. <Term> means …`), quoted fragments, anything sentence-shaped;
5. emits each table as its own chunk with the markdown grid intact, and keeps its text out
   of the prose stream so cells are not indexed twice;
6. **deduplicates** running furniture rather than deleting it — the first occurrence is
   kept, because the plan's UIN lives in the page header and was otherwise absent from the
   corpus entirely;
7. **falls back to overlapping windows** when structure is absent, because many policy
   PDFs are flat exports and the pipeline must not require structure to exist.

Every chunk carries the full metadata contract — tenant, document, policy, page range,
section path, chunk type, embedding model. Citations, isolation and evaluation all
depend on it.

---

## Security

| Control | Where |
|---|---|
| Tenant isolation | `tenant_id` on every table from migration 0001; all reads via `db.session.scoped`; adversarial test in `tests/test_tenant_isolation.py` |
| Untrusted source text | `rag/generation/prompt.py`; red-team fixture in `tests/test_prompt_injection.py` |
| Upload validation | MIME sniffed from content not extension, size cap, page cap, SHA-256 dedup |
| Citation integrity | `rag/validation/citations.py` |
| Audit trail | chunk ids, model and prompt version, latency and tokens on `messages` — not document text |

Authentication is stubbed to a fixed development tenant until week 5. **The boundary it
sits in front of is not stubbed** — that was the point of building it first.

---

## Evaluation

```bash
policy eval --set lic-real --mode hybrid_bm25 --k 5        # one configuration
policy experiments --set lic-real --rerank --generate      # the headline matrix
policy experiments --set acme-synthetic --k 1              # the regression fixture
policy eval-draft <policy_id> --out candidates.yaml
```

Two suites, deliberately separate. `lic-real` (47 questions, real LIC wordings) is the
headline benchmark and reports to `docs/experiments.md`. `acme-synthetic` (23 questions,
a generated 4-page document) is a deterministic regression fixture that runs **without the
real PDFs**, which matters because those are gitignored. Merging them would let an easy
corpus inflate the reported numbers.

`evals/questions.yaml` holds the frozen set. Labels are split by cost: page and section
labels are cheap and already unlock Recall@k, MRR, nDCG and the abstention metrics; gold
answers are expensive and lag behind.

**Evidence is labelled by page, not by chunk id.** That is deliberate, and this phase
proved the point: the chunker changed substantially, and the labels survived untouched.
Chunk-id labels would have been invalidated wholesale, quietly turning the benchmark into
a measure of whether chunk boundaries moved.

**Answerability is tri-state** — `yes`, `no`, `conditional`. The third state exists
because a boolean scored the brief's own example question wrong: *"Does this policy cover a
surgery after six months?"* depends on facts the question does not supply, so the system's
entirely defensible abstention was counted as an error. Conditional questions leave both
binary rates and are scored on whether **every** competing condition reached the model.

Two rules this project holds itself to: every experiment runs against the same frozen
set, and **the rows that lost stay in the table**. No accuracy figure appears in this
README that `policy experiments` did not produce.

`policy eval-draft` proposes candidate questions from an ingested policy, each tagged
with its source page and marked `UNVERIFIED`. It deliberately does **not** write the
benchmark: a set that labelled itself would measure the model's agreement with itself,
and every number built on it would be worthless.

---

## Layout

```
plan.md                  analysis of the brief + build sequence
rag/
  ingestion/             PyMuPDF spans, tables (bbox-aware), furniture dedup, pipeline
  chunking/              PART-anchored headings, section paths, table exclusion, fallback
  embeddings/            OpenAI embeddings, batched, model recorded per vector
  retrieval/
    types.py             RetrievedChunk: ranking score AND confidence score
    vector.py            exact tenant-scoped cosine search
    keyword.py           ts_rank_cd (named honestly -- this is not BM25)
    bm25.py              real BM25, experiment arm only (re-indexes per query)
    section.py           section-path arm -- built, measured, rejected
    fusion.py            reciprocal rank fusion
    rerank.py            cross-encoder, behind the [rerank] extra
    search.py            the one entry point API, CLI and harness all share
  generation/            prompt construction, structured output, bounded retry
  validation/            citation id validation
  evaluation/            dataset (tri-state), metrics, runner, report, drafting
  cli.py                 ingest / ask / policies / inspect / reindex / eval / experiments
services/api/            FastAPI routes, schemas, dependencies
db/                      models, session + scoping helper, Alembic migrations
tests/                   chunking, citations, generation, injection, isolation, API
evals/                   questions.yaml (synthetic) + questions-lic.yaml (headline)
                         results/<set>/  per-question output behind every table row
docs/                    decisions, experiments, sample-data provenance
```

---

## Limitations

- **Still two documents from one insurer, both pure-risk term plans.** No endowment,
  ULIP, annuity or health wording, and no second insurer. The `PART A..G` anchor is
  IRDAI-prescribed and *should* generalise across Indian insurers, but that is an
  expectation, not a measurement.
- **Table handling is largely unexercised.** These documents contain 2 tables across 17
  pages; term wordings are overwhelmingly prose. The phase-1 claim that tables were "the
  #1 technical risk" was mis-ranked for this corpus — health policies are where that work
  will actually be tested.
- **Retrieval serves vector-only** although hybrid measured better, because the hybrid
  that won depends on an in-memory BM25 arm that re-indexes per query. `pg_search` first,
  then re-measure, then change the default.
- **No true version-conflict test.** Two different products, not two versions of one.
  Attribution across near-duplicate wordings is tested; superseded-version behaviour is
  not.
- No OCR: a fully scanned PDF fails ingestion with an explicit error rather than
  indexing nothing silently. Neither real document needs it.
- The abstention threshold is still the initial 0.25 rather than a value calibrated
  against the set. It is at least now applied to a stable scale — see the score-scale
  entry in `docs/decisions.md`.
- No ANN index on embeddings. Correct and fast at 8 chunks; revisit at a few thousand.
- Single stubbed tenant. The isolation mechanism is real and tested; the authentication
  in front of it is not built.
- **This system is not a claims authority.** It describes what policy wording says.
  Every answer must be verified against the source wording by the agent before it
  reaches a customer.
