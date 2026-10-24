"""Ingestion pipeline: file -> chunks -> vectors -> database.

Runs in-process via FastAPI BackgroundTasks. A broker and worker process would be three
moving parts for a problem that does not exist at this volume; the ``status`` column on
``documents`` carries the progress a queue would otherwise report (plan.md §2.12).
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from db.models import Chunk, Document, DocumentStatus
from rag.chunking.chunker import chunk_document
from rag.config import get_settings
from rag.embeddings.openai_embeddings import embed_texts
from rag.ingestion.pdf import extract_document


class IngestionError(RuntimeError):
    pass


def ingest_document(session: Session, document_id: uuid.UUID, tenant_id: uuid.UUID) -> int:
    """Parse, chunk, embed and store one document. Returns the chunk count.

    Failures are recorded on the document row rather than swallowed: a document stuck in
    ``parsing`` with no error message is the worst possible outcome for debugging.
    """
    settings = get_settings()
    document = session.get(Document, document_id)
    if document is None or document.tenant_id != tenant_id:
        raise IngestionError("Document not found in this tenant")

    try:
        document.status = DocumentStatus.PARSING
        document.error = None
        session.flush()

        extracted = extract_document(Path(document.storage_path), max_pages=settings.max_pages)
        document.page_count = extracted.page_count

        if extracted.low_text_pages:
            # OCR is a week-2 item; for now say so loudly rather than indexing blank pages.
            if len(extracted.low_text_pages) == extracted.page_count:
                raise IngestionError(
                    "No extractable text on any page -- this looks like a scanned PDF. "
                    "OCR fallback is not implemented yet."
                )

        drafts = chunk_document(extracted)
        if not drafts:
            raise IngestionError("Chunking produced no chunks")

        session.query(Chunk).filter(
            Chunk.document_id == document.id, Chunk.tenant_id == tenant_id
        ).delete()

        document.status = DocumentStatus.EMBEDDING
        session.flush()

        embeddings = embed_texts([d.text for d in drafts])

        for draft, vector in zip(drafts, embeddings.vectors, strict=True):
            session.add(
                Chunk(
                    tenant_id=tenant_id,
                    document_id=document.id,
                    policy_id=document.policy_id,
                    ordinal=draft.ordinal,
                    page_from=draft.page_from,
                    page_to=draft.page_to,
                    section_path=draft.section_path,
                    chunk_type=draft.chunk_type,
                    text=draft.text,
                    token_count=draft.token_count,
                    embedding=vector,
                    embedding_model=embeddings.model,
                    embedding_dim=embeddings.dim,
                )
            )

        document.status = DocumentStatus.READY
        session.flush()
        return len(drafts)

    except Exception as exc:
        session.rollback()
        document = session.get(Document, document_id)
        if document is not None:
            document.status = DocumentStatus.FAILED
            document.error = str(exc)[:2000]
            session.commit()
        raise
