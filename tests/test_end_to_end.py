"""The week-1 slice end to end, against a real database.

OpenAI is stubbed in both places it is used -- embeddings and generation -- so this runs
without a key and without network. What it exercises is everything else: extraction,
chunking, the database write path, tenant-scoped retrieval, prompt construction,
citation validation, and page attribution surviving the whole journey.

Needs PostgreSQL:  docker compose up -d && alembic upgrade head && pytest -m db
"""

from __future__ import annotations

import uuid

import pytest

from db.models import DocumentStatus, Tenant
from rag.config import get_settings
from rag.generation.answer import generate_answer
from rag.retrieval.search import search
from rag.retrieval.types import RetrievalMode
from tests.conftest import DIM, fake_embed_one
from tests.test_generation import StubModel

pytestmark = pytest.mark.db


def test_ingestion_stores_chunks_and_marks_the_document_ready(ingested, session):
    tenant, _, document, count = ingested
    session.refresh(document)
    assert count > 0
    assert document.status == DocumentStatus.READY
    assert document.page_count == 4
    assert document.error is None


def test_retrieval_finds_the_waiting_period_clause(ingested, session):
    tenant, policy, _, _ = ingested
    question = "What is the waiting period for pre-existing diseases?"
    results = search(
        session,
        tenant.id,
        question,
        policy_ids=[policy.id],
        mode=RetrievalMode.VECTOR,
        query_vector=fake_embed_one(question),
        top_k=5,
    )
    assert results
    joined = " ".join(r.text for r in results)
    assert "thirty six months" in joined


def test_every_stored_vector_records_the_model_that_produced_it(ingested, session):
    """Without this, a model change silently mixes vector spaces (plan.md §2.9)."""
    from sqlalchemy import select

    from db.models import Chunk

    _, _, document, _ = ingested
    chunks = list(session.scalars(select(Chunk).where(Chunk.document_id == document.id)))
    assert chunks
    assert all(c.embedding_model == get_settings().openai_embedding_model for c in chunks)
    assert all(c.embedding_dim == DIM for c in chunks)
    assert all(c.embedding is not None for c in chunks)


def test_answer_citations_resolve_to_the_correct_page(ingested, session):
    """The whole point of the slice: a citation an agent can open the PDF and check."""
    tenant, policy, _, _ = ingested
    question = "What is the waiting period for pre-existing diseases?"
    chunks = search(
        session,
        tenant.id,
        question,
        policy_ids=[policy.id],
        mode=RetrievalMode.VECTOR,
        query_vector=fake_embed_one(question),
        top_k=5,
    )
    target = next(c for c in chunks if "thirty six months" in c.text)

    model = StubModel(
        [
            {
                "answerable": True,
                "answer": "Thirty six months of continuous coverage.",
                "claims": [
                    {
                        "text": "Thirty six months of continuous coverage.",
                        "citation_ids": [target.short_id],
                    }
                ],
                "insufficient_evidence_reason": None,
            }
        ]
    )
    answer = generate_answer(question, chunks, model=model)

    assert answer.answerable
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    # Page 2 of the fixture is where the waiting periods section actually is.
    assert citation.page_from == 2
    assert "Waiting Periods" in citation.section_path


def test_a_fabricated_citation_never_reaches_the_user(ingested, session):
    tenant, policy, _, _ = ingested
    chunks = search(
        session,
        tenant.id,
        "cosmetic surgery",
        policy_ids=[policy.id],
        mode=RetrievalMode.VECTOR,
        query_vector=fake_embed_one("cosmetic surgery"),
        top_k=5,
    )
    fabricated = {
        "answerable": True,
        "answer": "Covered in full from day one.",
        "claims": [
            {"text": "Covered in full from day one.", "citation_ids": ["c_00000000"]}
        ],
        "insufficient_evidence_reason": None,
    }
    answer = generate_answer("is it covered?", chunks, model=StubModel([fabricated, fabricated]))

    assert not answer.answerable
    assert answer.citations == []
    assert "Covered in full" not in answer.text


def test_retrieval_from_another_tenant_returns_nothing(ingested, session):
    """Same corpus, different tenant: the boundary holds."""
    _, policy, _, _ = ingested
    other = Tenant(slug=f"other-{uuid.uuid4().hex[:8]}", name="Other")
    session.add(other)
    session.flush()

    results = search(
        session,
        other.id,
        "waiting period pre-existing disease",
        mode=RetrievalMode.VECTOR,
        query_vector=fake_embed_one("waiting period pre-existing disease"),
        top_k=20,
    )
    assert results == []
