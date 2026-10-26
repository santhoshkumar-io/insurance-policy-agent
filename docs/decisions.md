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

---

## 2026-10-24 — Ranking score and confidence score are separate fields

**Decision.** `RetrievedChunk` carries both `score` (whatever the active strategy ranked
by) and `semantic_score` (cosine similarity, always populated). Abstention reads
`confidence`, which prefers `semantic_score`. The setting was renamed
`abstain_score_threshold` → `abstain_semantic_threshold`.

**Why.** The week-1 threshold of `0.25` was tuned against cosine similarity. A top RRF
score is about `1/61 ≈ 0.016` and a cross-encoder logit is unbounded and frequently
negative. Comparing either against `0.25` abstains on *everything* — so switching on
hybrid retrieval would have silently converted the system into one that refuses every
question, and every retrieval experiment would then have been measuring abstention
rather than retrieval.

Keyword-only results have no cosine score of their own, so `search()` computes one for
the chunks it keeps rather than leaving `None` to be read as zero confidence.

`tests/test_abstention_scale.py` asserts the same evidence produces the same abstention
decision under cosine, RRF, negative-logit and large-logit ranking.

**Observed live**, asking "What is the room rent limit and its waiting period?" against
the same chunk under two modes:

| Mode | Ranking score | Cosine | Threshold 0.25 |
|---|---|---|---|
| vector | 0.5432 | 0.543 | passes either way |
| hybrid | **0.0325** | 0.543 | **would have abstained on the ranking score** |

Same chunk, same correct answer, same evidence — and a ranking score 16x below the
threshold. The separation is what keeps that answer reaching the user.

**Reverses if.** Nothing. The old name is what invited the bug.

---

## 2026-10-24 — Table regions are excluded from the prose stream

**Decision.** `Table` carries its bbox; chunking drops any line falling inside one.

**Why.** PyMuPDF's span extraction returns table cell text along with everything else,
so every cell was being indexed twice — once as a markdown grid in the table chunk, and
once as linearised words in a prose chunk. The duplicate also explained the stray
section path `6. Table of Benefits >  > Benefit Limit Waiting period`: the table's bold
header row was prominent enough to pass for a heading.

Tables now also inherit the section path in force at their position, so a table citation
reads `6. Table of Benefits` instead of nothing — which is the difference between a
citation an agent can act on and one they have to go hunting for.

**Reverses if.** A document where `find_tables()` over-detects and swallows real prose.
The fallback is already there: no detected tables means no lines dropped.

---

## 2026-10-24 — The keyword arm is `ts_rank_cd`, and says so

**Decision.** `rag/retrieval/keyword.py` uses PostgreSQL full-text ranking. Real BM25
runs as a separate experiment row via `rank_bm25` in memory.

**Why.** Stock PostgreSQL has no BM25 — no document-length normalisation of the BM25
form, no k1/b. The brief's stack table says "BM25"; shipping `ts_rank_cd` under that name
is the kind of thing a reviewer catches, and it costs more credibility than the feature
gains. The BM25 row exists so the comparison is measured rather than asserted.

**Reverses if.** BM25 wins decisively on a real corpus. The answer then is the ParadeDB
`pg_search` extension, not the in-memory arm, which re-indexes per query and is honest
about being a comparison tool rather than a shipping mode.

---

## 2026-10-24 — Latency is measured with the embedding cached for every mode

**Decision.** The eval runner embeds each question once and reuses it across all
configurations, including the keyword-only ones.

**Why.** The first matrix run reported keyword retrieval at 323ms against vector's 12ms.
That gap was not retrieval: keyword-only mode skipped the cached embedding, so
`_fill_semantic_scores` made a live embedding call *inside the timed section*. Since the
confidence score is needed in every mode, the embedding is part of every request's cost
and belongs outside the comparison. Left unfixed, the table would have shown a 25x
latency penalty for keyword search that does not exist.

---

## 2026-10-24 — Default retrieval mode reverted to `vector`, on the evidence

**Decision.** `retrieval_mode` defaults to `vector`, not `hybrid`. Hybrid retrieval, BM25
and the cross-encoder all stay built and are one flag away.

**Why.** The whole point of building the benchmark was to let it decide, and it decided
against the more elaborate option. At k=1 over 23 questions (18 answerable):

| Configuration | Recall@1 | MRR | p50 |
|---|---|---|---|
| vector | **0.944** | **0.972** | 16ms |
| keyword (`ts_rank_cd`) | 0.833 | 0.917 | 4ms |
| bm25 (`rank_bm25`) | 0.889 | 0.921 | 4ms |
| hybrid RRF(vector + ts_rank_cd) | 0.889 | 0.944 | 5ms |
| hybrid RRF(vector + bm25) | 0.889 | 0.935 | 5ms |
| hybrid + cross-encoder | 0.944 | 0.972 | 27ms |

Hybrid retrieval is **worse** than vector alone at rank 1. The mechanism is visible in
the per-question output: RRF gives a chunk with a weak lexical overlap enough fused score
to displace the chunk the vector arm had ranked first. On a corpus where the questions are
paraphrases and the embeddings are good, the lexical arm contributes mostly noise.

The cross-encoder is the interesting row: it recovers exactly what the hybrid lost —
0.944/0.972, identical to plain vector — for 27ms instead of 16ms and a torch dependency.
So it is a good *repair* for fusion damage and not an improvement on the baseline. Both
stay off.

This is the brief's own stack principle applied literally: keep the simplest version that
meets the target, and add machinery only when the workload requires it.

**Reverses if.** Real insurer wordings change the picture, which they very well might —
see the caveat below. The expected failure mode for vector-only is exactly what the
keyword arm is for: literal lookups of clause numbers, defined terms and UINs. This
corpus has almost none, because it is four pages long.

**The caveat that matters.** This verdict rests on 18 answerable questions over one
synthetic 4-page document, where Recall@5 was 1.000 for every configuration — the
benchmark could not discriminate at all until k was dropped to 1. An MRR spread of 0.055
across six configurations is not a reliable ranking. The decision is "the evidence does
not justify the complexity *yet*", not "hybrid retrieval does not work".

---

## 2026-10-24 — The experiment report states its own limitations

**Decision.** `rag/evaluation/report.py` detects and prints its own saturation: when
every configuration scores identical recall, or the MRR spread falls inside the noise
floor, the generated table says so in the file.

**Why.** The first matrix run returned Recall@5 = 1.000 and section accuracy = 1.000 for
all six configurations. Read without context that looks like a solved retrieval problem;
it actually means the cut-off was wide relative to an 8-chunk corpus. A table that can be
misread that badly should not rely on the reader to catch it.

The generated block is fenced between markers so hand-written analysis around it survives
regeneration — the analysis is the part that makes the numbers mean anything.

---

## 2026-10-26 — Heading detection is anchored on the `PART` backbone

**Decision.** `PART A..G` is recognised explicitly and pins section level 1; nothing else
may occupy that level. Numbered clauses start at level 2.

**Why.** The font signal that works on a clean document barely exists in real IRDAI-filed
wordings. In LIC's Digi Term, 564 body lines and 138 heading-ish lines are **all 10.0pt** —
headings are the same point size as body text, and bold is used for mid-sentence emphasis
too. Prominence alone accepted 71 lines as headings where about 53 were real.

What is reliable is the filed structure: `PART- A`, `PART– B: DEFINITIONS`,
`PART– C: BENEFITS`, `PART – D: CONDITIONS RELATED TO SERVICING ASPECTS`, `PART E`,
`PART – F`, `PART – G`. Identical in both documents, because IRDAI prescribes it. That is
a domain rule, not a per-document hack.

Pinning level 1 was the other half of the fix. A numbered clause claiming level 1 wiped
out the parent for every chunk that followed, which is how `1. Proof of Age:` lost
`PART – D` and a stray bold `Name` on the schedule page displaced `PART- A`.

Four more fixes landed with it, each with an observed failure behind it: line size is now
the dominant span size rather than the max (a bold `18.` was promoting whole definition
sentences to level-1 headings); purely numeric candidates are rejected (17 of 71 accepted
headings were page numbers); numeric furniture is stripped by y-band because `1`, `2`, `3`
never match each other by text; and empty padding levels are dropped from the rendered
path, which is what produced `` >  > PART- A``.

Result: every section path in both documents now sits under its correct PART, and reads
like the document's actual structure.

**Reverses if.** A corpus of wordings that are not IRDAI-filed. The generic rules still
apply; only the PART anchor is India-specific.

---

## 2026-10-26 — Running furniture is deduplicated, not deleted

**Decision.** `_strip_running_headers` keeps the earliest occurrence of each repeated
line and strips the rest.

**Why.** Found by a benchmark question rather than by a test. *"What is the UIN of this
plan?"* was unanswerable, and not because the document omits it: `512N356V02` sits in the
running header of all 17 pages, so furniture detection removed it from the corpus
**entirely**. The insurer name and the plan name had the same problem.

These lines are genuinely repeated page furniture *and* genuinely substantive. Keeping
the title-page occurrence preserves the information exactly once, where it is most
meaningful, and still removes the noise from the other sixteen pages. Page numbers remain
deleted outright — they carry no information even once.

This is the clearest case so far of the benchmark earning its cost: no unit test would
have caught content that was silently absent.

**Reverses if.** Nothing. The previous behaviour was destroying data.

---

## 2026-10-26 — Real BM25 beats `ts_rank_cd` decisively, and the reversal condition has fired

**Decision.** The phase-2 entry on the keyword arm said it would reverse "if BM25 wins
decisively on a real corpus — the answer then is the ParadeDB `pg_search` extension".
On the LIC corpus at k=5 over 41 answerable questions:

| Arm | Recall@5 | MRR |
|---|---|---|
| `ts_rank_cd` | 0.872 | 0.608 |
| real BM25 (`rank_bm25`) | **0.974** | **0.795** |
| hybrid RRF(vector + `ts_rank_cd`) | 0.897 | 0.782 |
| hybrid RRF(vector + BM25) | **1.000** | 0.898 |
| vector alone | 0.949 | 0.911 |

An MRR gap of 0.187 between the two lexical arms is not noise, and on the synthetic
corpus it was 0.032. **So `pg_search` is now justified** and goes on the phase-4 list.

**Why the gap is real.** BM25 weights rare terms by inverse document frequency;
`ts_rank_cd` does not. These documents are full of rare, high-value tokens — `512N356V02`,
`Section 45`, `Unexpired Risk Premium Value`, `Panchnama` — and the synthetic corpus had
almost none, which is exactly the limitation phase 2 recorded.

---

## 2026-10-26 — Hybrid retrieval wins on a real corpus: the phase-2 verdict was corpus-bound

**Decision.** `retrieval_mode` stays `vector` for now, but the phase-2 conclusion that
"plain vector beats every hybrid" is **withdrawn** as a general claim. It held only on an
8-chunk synthetic document.

`hybrid RRF(vector + BM25)` reaches **Recall@5 = 1.000** against vector's 0.949 — every
answerable question's evidence retrieved, which vector alone does not manage. Vector
retains a narrow MRR lead (0.911 vs 0.898), and `hybrid + cross-encoder` has the best MRR
of all (0.932).

**Why the default has not changed yet.** The configuration that wins on recall uses the
in-memory BM25 arm, which re-indexes the whole corpus on every query. That is a benchmark
tool, not a serving path, and shipping it would trade a correctness win for a latency
problem that grows with the corpus. The honest sequence is `pg_search` first, then
re-measure, then switch the default to a hybrid that is actually deployable.

Recording this rather than quietly flipping a setting is the point: the measurement says
hybrid is better, and the engineering says the version that measured better cannot ship.

---

## 2026-10-26 — Section-aware retrieval was my own top recommendation, and it lost

**Decision.** `HYBRID_SECTION` and `HYBRID_BM25_SECTION` are built, tested and **not
adopted**. `rag/retrieval/section.py` stays in the tree behind a flag.

**Why it was built.** Phase 2's sharpest finding: every configuration returned the
*definition* of a pre-existing disease above the *waiting-period clause*, because the two
passages share nearly all their vocabulary, and a cross-encoder shown both still preferred
the definition. The section path is what distinguishes them, and it was unused at query
time. I called it "the first thing to try in the next phase".

**What the measurement said:**

| Configuration | Recall@5 | MRR |
|---|---|---|
| hybrid RRF(vector + `ts_rank_cd`) | 0.897 | 0.782 |
| + section arm | 0.923 | 0.772 |
| hybrid RRF(vector + BM25) | **1.000** | **0.898** |
| + section arm | 0.897 | 0.750 |

Added to the weaker lexical pair it buys a little recall for a little ranking. Added to
the best pair it is **actively destructive** — 10 points of recall and 15 of MRR. The
synthetic corpus agrees (0.889 → 0.722 at k=1), so this is not a one-corpus artifact.

**Why it fails, which is the useful part.** A section path identifies a *region*, not a
*passage*. Eight chunks share `PART – D: CONDITIONS RELATED TO SERVICING ASPECTS`, so a
query matching that path returns all eight at high rank and each contributes its
reciprocal to the fusion. The arm floods RRF with weakly-relevant candidates and dilutes
the genuinely relevant ones. RRF treats every arm as an equal voter, and this signal is
not of equal quality.

The diagnosis was right and the remedy was wrong. Section metadata should be a
**tie-breaker or a filter**, not a voting arm — re-rank within the fused list, or restrict
candidates when a query clearly names a section. That is worth trying; adding it as a
fourth voter is not.

**Reverses if.** Reimplemented as a tie-breaker and measured again.
