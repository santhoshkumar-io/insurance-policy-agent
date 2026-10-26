"""Retrieval ranked on the section path.

Phase 2's sharpest failure: asked *"What is the waiting period for pre-existing
diseases?"*, every configuration returned the **definition** of a pre-existing disease
above the **waiting-period clause**. The two passages share nearly all their vocabulary
-- "pre-existing", "disease", "policy", "months", "insurer" -- so cosine similarity
preferred the denser match, and a cross-encoder shown both still preferred it.

What separates them is not in the body text at all. It is the address:

    1. Definitions > 1.1 Pre-existing Disease
    4. Waiting Periods > 4.3 Pre-existing disease waiting period

This arm ranks on that address alone, so "waiting period" in the question can pull the
clause above the definition. It is a third RRF arm rather than a score bonus: it needs no
weight to tune, and it earns its own row in the experiment table so it can be rejected on
evidence like everything else.

It depends entirely on section paths being correct, which is why the chunker repair came
first. Against the real LIC documents this arm would have been ranking on
`` >  > PART- A`` and whole definition sentences.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session

from db.models import Chunk
from rag.config import get_settings
from rag.retrieval.keyword import to_tsquery_text
from rag.retrieval.types import RetrievedBy, RetrievedChunk
from rag.retrieval.vector import base_statement, build_chunk


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
