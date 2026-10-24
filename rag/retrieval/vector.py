"""Exact, tenant-scoped vector retrieval.

No ANN index: every query here is filtered by tenant and policy set, which is precisely
the case where a pre-filtered HNSW traversal loses recall. At MVP corpus size exact
search is fast and correct by construction (plan.md §2.4).
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Chunk, Document, Policy
from db.session import scoped
from rag.config import get_settings
from rag.retrieval.types import RetrievedBy, RetrievedChunk


def base_statement(tenant_id: uuid.UUID, policy_ids: list[uuid.UUID] | None):
    """The scoped candidate set every strategy starts from.

    Both arms share this so that tenant isolation and the embedding-model guard cannot
    drift apart between strategies -- a hole in one arm is a hole in the system.
    """
    settings = get_settings()
    stmt = (
        select(Chunk, Document.filename, Policy.product_name)
        .join(Document, Document.id == Chunk.document_id)
        .join(Policy, Policy.id == Chunk.policy_id)
        .where(Chunk.embedding.isnot(None))
        # Refuse to mix embedding spaces in one ranking.
        .where(Chunk.embedding_model == settings.openai_embedding_model)
    )
    stmt = scoped(stmt, Chunk, tenant_id)
    if policy_ids:
        stmt = stmt.where(Chunk.policy_id.in_(policy_ids))
    return stmt


def build_chunk(chunk: Chunk, filename: str, product_name: str, **kwargs) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.id,
        text=chunk.text,
        page_from=chunk.page_from,
        page_to=chunk.page_to,
        section_path=chunk.section_path,
        chunk_type=chunk.chunk_type,
        policy_id=chunk.policy_id,
        document_id=chunk.document_id,
        document_name=filename,
        policy_name=product_name,
        **kwargs,
    )


def retrieve(
    session: Session,
    tenant_id: uuid.UUID,
    query_vector: list[float],
    policy_ids: list[uuid.UUID] | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Rank by cosine similarity to ``query_vector``."""
    top_k = top_k or get_settings().retrieval_top_k

    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        base_statement(tenant_id, policy_ids)
        .add_columns(distance)
        .order_by(distance)
        .limit(top_k)
    )

    results: list[RetrievedChunk] = []
    for rank, (chunk, filename, product_name, dist) in enumerate(
        session.execute(stmt).all(), start=1
    ):
        # cosine distance -> similarity, for a threshold that reads naturally.
        similarity = 1.0 - float(dist)
        results.append(
            build_chunk(
                chunk,
                filename,
                product_name,
                score=similarity,
                semantic_score=similarity,
                rank=rank,
                retrieved_by=RetrievedBy.VECTOR,
            )
        )
    return results


def semantic_scores_for(
    session: Session,
    tenant_id: uuid.UUID,
    query_vector: list[float],
    chunk_ids: list[uuid.UUID],
) -> dict[uuid.UUID, float]:
    """Cosine similarity for specific chunks, whatever found them.

    A chunk surfaced only by the keyword arm has no similarity score, but abstention
    still needs one on a comparable scale -- so it is computed here rather than left
    None and silently treated as zero confidence.
    """
    if not chunk_ids:
        return {}

    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        select(Chunk.id, distance)
        .where(Chunk.id.in_(chunk_ids), Chunk.embedding.isnot(None))
    )
    stmt = scoped(stmt, Chunk, tenant_id)
    return {row[0]: 1.0 - float(row[1]) for row in session.execute(stmt).all()}
