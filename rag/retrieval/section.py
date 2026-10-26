"""Retrieval ranked on the section path.

Phase 2's sharpest failure: asked *"What is the waiting period for pre-existing
diseases?"*, every configuration returned the **definition** of a pre-existing disease
above the **waiting-period clause**. The two passages share nearly all their vocabulary
-- "pre-existing", "disease", "policy", "months", "insurer" -- so cosine similarity
preferred the denser match, and a cross-encoder shown both still preferred it.

What separates them is not in the body text at all. It is the address:

    1. Definitions > 1.1 Pre-existing Disease
    4. Waiting Periods > 4.3 Pre-existing disease waiting period

It was first built as a third RRF arm, and **that lost** -- on the real corpus it cost the
best configuration 10 points of recall and 15 of MRR. The reason is instructive: a section
path identifies a *region*, not a passage. Eight chunks share
``PART – D: CONDITIONS RELATED TO SERVICING ASPECTS``, so a query matching that path
returns all eight at high rank and each contributes its reciprocal to the fusion. The arm
floods RRF with weakly-relevant candidates and dilutes the ones that matter, because RRF
treats every arm as an equal voter and this signal is not of equal quality.

So the signal is used as a **tie-breaker** instead. :func:`apply_tiebreak` reorders results
that fusion already scored near-equally and **never adds a candidate**, which is precisely
the failure mode of the arm version. The diagnosis was right; only the remedy was wrong.

Everything here depends on section paths being correct, which is why the chunker repair
came first. Against the real LIC documents this would have been ranking on
`` >  > PART- A`` and whole definition sentences.
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session

from db.models import Chunk
from rag.config import get_settings
from rag.retrieval.keyword import to_tsquery_text
from rag.retrieval.types import RetrievedBy, RetrievedChunk
from rag.retrieval.vector import base_statement, build_chunk

# Words worth matching against a section path; numbers and punctuation are noise
# there, since section paths are titles.
_TERM = re.compile(r"[A-Za-z][A-Za-z0-9-]*")


def retrieve(
    session: Session,
    tenant_id: uuid.UUID,
    question: str,
    policy_ids: list[uuid.UUID] | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or get_settings().retrieval_top_k

    query_text = to_tsquery_text(question)
    if not query_text:
        return []

    tsquery = func.to_tsquery("english", query_text)
    rank = func.ts_rank_cd(Chunk.section_tsv, tsquery).label("section_rank")

    stmt = (
        base_statement(tenant_id, policy_ids)
        .add_columns(rank)
        .where(Chunk.section_tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(top_k)
    )

    results: list[RetrievedChunk] = []
    for position, (chunk, filename, product_name, score) in enumerate(
        session.execute(stmt).all(), start=1
    ):
        results.append(
            build_chunk(
                chunk,
                filename,
                product_name,
                score=float(score),
                # Not comparable to cosine similarity; search() supplies the confidence
                # score for whatever it keeps.
                semantic_score=None,
                rank=position,
                retrieved_by=RetrievedBy.SECTION,
            )
        )
    return results


def matches_section(question: str, chunk: RetrievedChunk) -> int:
    """How many query terms appear in this chunk's section path."""
    path = (chunk.section_path or "").lower()
    if not path:
        return 0
    terms = {t for t in tokenize_terms(question) if len(t) > 3}
    return sum(1 for term in terms if term in path)


def tokenize_terms(question: str) -> list[str]:
    return [t.lower() for t in _TERM.findall(question)]


def apply_tiebreak(
    question: str, results: list[RetrievedChunk], tolerance: float | None = None
) -> list[RetrievedChunk]:
    """Reorder near-equally-scored results so section matches come first.

    The contract that distinguishes this from the arm version:

    * it never adds a candidate and never drops one -- the output is a permutation of the
      input, asserted in tests;
    * it only reorders *within* a tolerance band, so it cannot override a genuine score
      difference. A chunk fusion ranked clearly higher stays higher.

    ``tolerance`` is relative to the better score of the pair being compared, so it works
    unchanged across the scales the different strategies produce -- cosine similarity, an
    RRF sum around 0.03, or a cross-encoder logit.
    """
    if len(results) < 2:
        return results

    if tolerance is None:
        tolerance = get_settings().section_tiebreak_tolerance

    # Group consecutive results whose scores are within the band of the group's leader.
    groups: list[list[RetrievedChunk]] = []
    for chunk in results:
        if groups:
            leader = groups[-1][0]
            reference = abs(leader.score) or 1.0
            if abs(leader.score - chunk.score) <= tolerance * reference:
                groups[-1].append(chunk)
                continue
        groups.append([chunk])

    reordered: list[RetrievedChunk] = []
    for group in groups:
        if len(group) > 1:
            # Stable sort: equal section-match counts keep fusion's original order.
            group = sorted(group, key=lambda c: matches_section(question, c), reverse=True)
        reordered.extend(group)

    for position, chunk in enumerate(reordered, start=1):
        chunk.rank = position
    return reordered
