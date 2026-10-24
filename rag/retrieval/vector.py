"""Exact, tenant-scoped vector retrieval.

No ANN index: every query here is filtered by tenant and policy set, which is precisely
the case where a pre-filtered HNSW traversal loses recall. At MVP corpus size exact
search is fast and correct by construction (plan.md §2.4).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Chunk, Document, Policy
from db.session import scoped
from rag.config import get_settings


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    text: str
    page_from: int
    page_to: int
    section_path: str
    chunk_type: str
    policy_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    policy_name: str
    score: float

    @property
    def short_id(self) -> str:
        """The handle the model is allowed to cite. Short enough not to waste tokens."""
        return f"c_{str(self.chunk_id)[:8]}"


def retrieve(
    session: Session,
    tenant_id: uuid.UUID,
    query_vector: list[float],
    policy_ids: list[uuid.UUID] | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k

    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")

    stmt = (
        select(Chunk, Document.filename, Policy.product_name, distance)
        .join(Document, Document.id == Chunk.document_id)
        .join(Policy, Policy.id == Chunk.policy_id)
        .where(Chunk.embedding.isnot(None))
        # Refuse to mix embedding spaces in one ranking.
        .where(Chunk.embedding_model == settings.openai_embedding_model)
        .order_by(distance)
        .limit(top_k)
    )
    stmt = scoped(stmt, Chunk, tenant_id)
    if policy_ids:
        stmt = stmt.where(Chunk.policy_id.in_(policy_ids))

    results: list[RetrievedChunk] = []
    for chunk, filename, product_name, dist in session.execute(stmt).all():
        results.append(
            RetrievedChunk(
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
                # cosine distance -> similarity, for a threshold that reads naturally.
                score=1.0 - float(dist),
            )
        )
    return results
