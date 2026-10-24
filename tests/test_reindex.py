"""Re-indexing: re-chunk and re-embed in place.

Without this, improving the chunker means deleting and re-uploading every document,
which is how a corpus ends up half-indexed by an old strategy and nobody notices.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from db.models import Chunk, Document, DocumentStatus
from rag.ingestion.pipeline import ingest_document

pytestmark = pytest.mark.db


def chunk_count(session, document_id) -> int:
    return int(
        session.scalar(select(func.count(Chunk.id)).where(Chunk.document_id == document_id)) or 0
    )


def test_reindex_replaces_chunks_instead_of_duplicating_them(ingested, session):
    tenant, _, document, first_count = ingested
    assert chunk_count(session, document.id) == first_count

    second_count = ingest_document(session, document.id, tenant.id)

    assert second_count == first_count
    assert chunk_count(session, document.id) == first_count, "re-index duplicated chunks"


def test_reindex_leaves_the_document_ready(ingested, session):
    tenant, _, document, _ = ingested
    ingest_document(session, document.id, tenant.id)
    session.refresh(document)
    assert document.status == DocumentStatus.READY
    assert document.error is None


def test_reindex_repopulates_embeddings_and_their_model(ingested, session):
    from rag.config import get_settings

    tenant, _, document, _ = ingested
    ingest_document(session, document.id, tenant.id)

    chunks = list(session.scalars(select(Chunk).where(Chunk.document_id == document.id)))
    assert chunks
    assert all(c.embedding is not None for c in chunks)
    assert all(c.embedding_model == get_settings().openai_embedding_model for c in chunks)


def test_reindex_refuses_a_document_from_another_tenant(ingested, session):
    """The re-index path takes an id from the caller, so it checks the tenant itself."""
    import uuid

    from db.models import Tenant
    from rag.ingestion.pipeline import IngestionError

    _, _, document, _ = ingested
    other = Tenant(slug=f"ri-{uuid.uuid4().hex[:8]}", name="Other")
    session.add(other)
    session.flush()

    with pytest.raises(IngestionError):
        ingest_document(session, document.id, other.id)


def test_reindex_of_a_missing_document_raises(session):
    import uuid

    from rag.ingestion.pipeline import IngestionError

    with pytest.raises(IngestionError):
        ingest_document(session, uuid.uuid4(), uuid.uuid4())


def test_document_row_survives_reindex(ingested, session):
    """The id is what citations and the policy library point at; it must be stable."""
    tenant, _, document, _ = ingested
    original_id, original_checksum = document.id, document.checksum
    ingest_document(session, document.id, tenant.id)
    refreshed = session.get(Document, original_id)
    assert refreshed is not None
    assert refreshed.checksum == original_checksum
