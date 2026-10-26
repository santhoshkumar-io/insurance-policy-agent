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
| Real BM25, cached per tenant + policy scope | ✅ **the default** — best recall, 4.9ms warm |
| Cross-encoder reranking | ✅ built, off by default; best MRR, costs recall |
| Section metadata as a ranking signal | ✅ built twice, **rejected twice on measurement** |
| Evaluation harness: Recall@k, MRR, nDCG, abstention, conditional coverage, cost | ✅ |
| Tri-state answerability (yes / no / conditional) | ✅ |
| Two separate benchmark suites (real + synthetic regression) | ✅ |
| 48-question benchmark on real LIC wordings | ✅ |
| Re-index without re-upload | ✅ |
| Candidate question drafting for the benchmark | ✅ |
| Structure-aware chunking of real IRDAI-filed wordings | ✅ |
| Next.js dashboard: upload, library, ask, citation panel | ✅ |
| PDFs stored in Postgres, so documents survive a redeploy | ✅ |
| Docker image, migrations and demo seeding on boot | ✅ |
| CI: lint, unit tests, and the db suite against pgvector | ✅ |
| Rate limits and a daily cap for an open demo | ✅ |
| Repeated-run generation metrics (`--repeats`, with a spread) | ✅ |
| BM25 inside Postgres (ParadeDB `pg_search`) | ruled out — unavailable on managed Postgres |
| OCR for scanned pages | not needed — both real documents extract cleanly |
| Answerability judgement in generation | **the next target** — see the benchmark section |
| Policy comparison | phase 5 |
| Real auth, tracing | phase 6 |
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

Or run the whole system in containers, which is what gets deployed:

```bash
docker compose up --build                 # migrates, seeds a demo policy, serves on :8000
```

---

## What the benchmark says

48 questions (38 answerable, 2 conditional, 8 unanswerable) hand-verified against two real
LIC policy wordings — *Digi Term* and *Jeevan Raksha*, 17 pages and 49 chunks each. Full
table, method and caveats in [docs/experiments.md](docs/experiments.md).

| Configuration | Recall@5 | MRR |
|---|---|---|
| vector | 0.950 | 0.913 |
| keyword (`ts_rank_cd`) | 0.850 | 0.593 |
| bm25 (`rank_bm25`) | 0.975 | 0.781 |
| hybrid RRF(vector + `ts_rank_cd`) | 0.875 | 0.766 |
| **hybrid RRF(vector + bm25)** — the default | **1.000** | 0.884 |
| hybrid RRF(vector + bm25) + section tie-break | 1.000 | 0.746 |
| hybrid + cross-encoder | 0.950 | **0.921** |

Rows that lost stay in the table. Two more losing configurations — the section arm in both
hybrids — were measured, removed from the code, and are kept in
[docs/experiments.md](docs/experiments.md) with the numbers that rejected them.

**Four results worth reading the detail for.**

**The previous phase's verdict was wrong, and the caveat that said so was the useful
part.** Phase 2 concluded plain vector beat every hybrid — on an 8-chunk synthetic
document, while stating that could not generalise. On a real corpus
`hybrid RRF(vector + bm25)` retrieves the evidence for **every** answerable question
(1.000) where vector alone does not (0.950). That conclusion is withdrawn.

**Real BM25 beats `ts_rank_cd` by 0.188 MRR** — against 0.032 on the synthetic corpus,
which was reported as noise at the time. The mechanism is IDF, and these documents are
dense with the rare tokens that rewards: `512N356V02`, `Section 45`, `Unexpired Risk
Premium Value`. Running BM25 as a measured experiment instead of asserting `ts_rank_cd`
was adequate is what made that visible, and it fired a reversal condition recorded in
[docs/decisions.md](docs/decisions.md).

**The winning configuration could not ship, so it was made shippable.** The BM25 arm
re-indexed the whole scoped corpus on *every* query — a benchmark tool, as its own
docstring said, which is why the default stayed `vector` for a phase despite the
measurement. It now builds one index per `(tenant, policy scope)` and reuses it:
**67.8ms cold, 4.9ms warm, a 13.7x speedup**, with invalidation by a fingerprint query
rather than a hook, because ingestion runs in another task and possibly another process.
ParadeDB `pg_search` would be the better long-term answer and is **not available on
managed Postgres**, which is the deployment target — so it stays unbuilt, and the trigger
for revisiting it is written down.

**The default is now `hybrid_bm25`, chosen on recall rather than MRR.** The reranked rows
have the best MRR (0.921) and lose 5 points of recall for 10x the latency. Recall decides
because the model is handed every one of the `top_k` chunks: evidence that never arrives
cannot be recovered by any later stage, while imperfect ordering inside that set is a soft
cost. An incidental finding — the two reranked rows are the same row to three decimals on
recall, MRR and section accuracy, so **the cross-encoder erases the difference between the
lexical arms** it was fed.

**Retrieval is finished; every remaining error is in generation.** The five questions
scored as false abstentions all have **`recall = 1.0`** — three with the evidence at rank
1. The text was retrieved, served to the model, and the model declined anyway. Asked
whether a loan is available, it answered "the sources do not provide any information"
while holding the sentence *"No loan will be available under this policy"*; asked whether
the death benefit option can be changed, it said the same while holding *"The Death
Benefit Option once chosen cannot be changed later"*. Both abstain **10 times out of 10**,
so this is reproducible, not sampling noise. Abstention is a feature of this system, and a
feature that fires when it should not is a defect with good manners — so the next target
is the answerability judgement, not retrieval.

### Section-aware retrieval was my top recommendation, and it lost

Phase 2 ended by naming this "the first thing to try in the next phase". Built, measured,
rejected twice. As a fourth RRF arm it cost the best pair **10 points of recall and 15 of
MRR**, and the synthetic corpus agreed. The failure analysis pointed at a tie-breaker that
reorders near-equal results and never introduces candidates, so that was built and
measured too — and a tolerance sweep found it **monotonically harmful** (MRR 0.884 off,
0.841 at 0.05, 0.746 at 0.15, recall flat at 1.000). Narrow enough to be safe is narrow
enough to do nothing.

Why it fails is the useful part. A section path identifies a *region*, not a *passage* —
eight chunks share `PART – D: CONDITIONS RELATED TO SERVICING ASPECTS`, so a query
matching that path returns all eight at high rank and each contributes its reciprocal to
the fusion. RRF treats every arm as an equal voter and this signal is not of equal
quality. The tie-breaker fails for the same reason in a different shape: a path cannot
distinguish "in the right section" from "is the right passage", and ranking needs the
second. Recorded as a dead end for ranking after two attempts; the remaining variant — a
hard filter when a query unambiguously names a section — is noted as untested, not
promised. `rag/retrieval/section.py` stays behind a flag.

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

## The dashboard

```bash
uvicorn services.api.main:app --reload     # terminal 1
cd apps/web && npm install && npm run dev  # terminal 2 -- http://localhost:3000
```

Three screens — upload, library, ask — and the citation panel is the point of all three.
For every chunk the model was served it shows the **page and section path read from the
database row**, never from the model's output, so a plausible invented page number has no
path to the screen. It shows which arm found the chunk, the retrieved text itself, and
**both scores side by side**: the ranking score, whose scale depends on the retrieval
mode, next to the cosine, which does not and which is what the abstention threshold reads.
Those being one unlabelled number was a real bug here, and the panel is where that
distinction stops being a paragraph in a decision log.

Chunks the model was shown but did not cite are labelled *retrieved, not cited*, because
what the model declined to lean on is as informative as what it used. And an abstention
renders **as an abstention, with its reason** — a blank answer box would undo the feature.

`npm run smoke` drives all three screens in a real browser against a running API and
checks what matters: that a citation card carries a page, both scores, and two *different*
numbers; that an abstention renders its reason; and that the console is clean. It is not
in CI — it needs a database, an ingested policy and an API key — but it found two defects
the build was happy with.

---

## Deploying it

Neon (Postgres + pgvector), Render (the API's `Dockerfile`), Vercel (the dashboard).
Step by step in [docs/deployment.md](docs/deployment.md).

Two things in that setup are decisions rather than configuration.

**Uploaded PDFs are stored in Postgres**, not on disk. The API used to write them to a
local directory, which works on a laptop and fails on every managed container platform:
the database survives a redeploy and the filesystem does not. The failure is quiet —
the library still lists the document and chat still answers from the chunks, and only
`policy reindex` breaks, which is to say the failure surfaces the first time the chunker
improves. `tests/test_storage.py` deletes the file and re-indexes, so the case is
reproduced rather than described. This is right for ~1MB wordings and wrong in the tens
of GB; the trigger to move to object storage is written down, and `rag/storage.py` is the
entire surface that would change.

**The demo is open, with ceilings instead of a login.** A per-IP sliding window held in
process, and a global daily cap counted from the `messages` table — because an in-process
counter resets on restart and is per worker, so N workers would authorise N times the
intended spend. `X-Forwarded-For` is caller-controlled and therefore evadable, which is
exactly why the limit that protects the bill lives in the database rather than in memory.

---

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
best **cosine** similarity below `ABSTAIN_SEMANTIC_THRESHOLD`; or the model reporting
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

Two suites, deliberately separate. `lic-real` (48 questions, real LIC wordings) is the
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

**Generation metrics are repeated, retrieval metrics are not.** `--repeats N` runs the
generation pass N times per question against the same retrieved evidence and reports the
mean rate with half the range across passes. One benchmark question was measured
answering 5 times and abstaining 5 out of 10 — at temperature 0, from identical evidence
— so a single pass cannot distinguish a real improvement from the coin landing
differently. Retrieval runs once, because it is deterministic.

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
    bm25.py              real BM25, cached per (tenant, policy scope), fingerprinted
    section.py           section metadata -- built as arm and as tie-break, both rejected
    fusion.py            reciprocal rank fusion
    rerank.py            cross-encoder, behind the [rerank] extra
    search.py            the one entry point API, CLI and harness all share
  generation/            prompt construction, structured output, bounded retry
  validation/            citation id validation
  evaluation/            dataset (tri-state), metrics, runner, report, drafting
  cli.py                 ingest / ask / policies / inspect / reindex / eval / experiments
services/api/            FastAPI routes, schemas, dependencies, CORS
apps/web/                Next.js dashboard: upload, library, ask + citation panel
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
- **The BM25 index lives in each worker's memory.** Cached per tenant and policy scope
  and invalidated by a fingerprint query, so it is correct and fast at ~100 chunks — but
  memory and cold start multiply by worker count, and this needs revisiting well before
  ~10⁵ chunks. `pg_search` is the answer then, and it requires self-hosted Postgres.
- **The system over-abstains.** Five benchmark questions whose evidence was retrieved at
  rank 1 were declined anyway, reproducibly. Retrieval is at Recall@5 = 1.000; the
  remaining errors are all in the answerability judgement.
- **No true version-conflict test.** Two different products, not two versions of one.
  Attribution across near-duplicate wordings is tested; superseded-version behaviour is
  not.
- No OCR: a fully scanned PDF fails ingestion with an explicit error rather than
  indexing nothing silently. Neither real document needs it.
- The abstention threshold is still the initial 0.25 rather than a value calibrated
  against the set. It is at least now applied to a stable scale — see the score-scale
  entry in `docs/decisions.md`.
- No ANN index on embeddings. Correct and fast at ~100 chunks; revisit at a few thousand.
- Single stubbed tenant, and the deployed demo is **open on purpose**. The isolation
  mechanism is real and tested — including in the BM25 cache key, where a miss would
  serve one tenant's index to another — but the authentication in front of it is not
  built, and the dashboard has no sign-in. A per-IP window and a global daily cap stand
  in for it, and the per-IP half is evadable by anyone who rotates `X-Forwarded-For`.
- PDFs live in a `bytea` column. Correct at this size, wrong in the tens of GB, and
  `SELECT *` on `documents` now carries the file with it.
- The dashboard has no automated test in CI. `npm run smoke` is a real browser check but
  needs a live API, a database and a key, so nothing guards the UI on a pull request.
- **This system is not a claims authority.** It describes what policy wording says.
  Every answer must be verified against the source wording by the agent before it
  reaches a customer.
