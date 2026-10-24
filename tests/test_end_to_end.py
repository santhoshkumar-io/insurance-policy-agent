"""The week-1 slice end to end, against a real database.

OpenAI is stubbed in both places it is used -- embeddings and generation -- so this runs
without a key and without network. What it exercises is everything else: extraction,
chunking, the database write path, tenant-scoped retrieval, prompt construction,
citation validation, and page attribution surviving the whole journey.

Needs PostgreSQL:  docker compose up -d && alembic upgrade head && pytest -m db
"""

from __future__ import annotations

import hashlib
import math
import uuid

import pytest

from db.models import Document, DocumentStatus, Policy, Tenant
from rag.config import get_settings
from rag.embeddings.openai_embeddings import EmbeddingResult
from rag.generation.answer import generate_answer
from rag.ingestion.pipeline import ingest_document
from rag.retrieval.vector import retrieve
from tests.test_generation import StubModel

pytestmark = pytest.mark.db

DIM = 1536


def fake_embed_one(text: str) -> list[float]:
    """A hashed bag-of-words embedding.

    Deterministic and offline, but genuinely lexical: a query about waiting periods
    really does land nearest the waiting-period clause, so retrieval is being tested
    rather than bypassed.
    """
    vector = [0.0] * DIM
    for word in text.lower().split():
        word = word.strip(".,;:()\"'")
        if not word:
            continue
        index = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
        vector[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


@pytest.fixture
def stub_openai(monkeypatch):
    settings = get_settings()

    def fake_embed_texts(texts):
        return EmbeddingResult(
            vectors=[fake_embed_one(t) for t in texts],
            model=settings.openai_embedding_model,
            dim=DIM,
        )

    monkeypatch.setattr("rag.ingestion.pipeline.embed_texts", fake_embed_texts)
    return fake_embed_texts


@pytest.fixture
def session():
    from db.session import get_session_factory

    session = get_session_factory()()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def ingested(session, stub_openai, structured_policy_pdf):
    tenant = Tenant(slug=f"e2e-{uuid.uuid4().hex[:8]}", name="E2E")
    session.add(tenant)
    session.flush()

    policy = Policy(tenant_id=tenant.id, insurer="Acme", product_name="Health Guard")
    session.add(policy)
    session.flush()

    document = Document(
        tenant_id=tenant.id,
        policy_id=policy.id,
        filename="structured_policy.pdf",
        storage_path=str(structured_policy_pdf),
        checksum=uuid.uuid4().hex * 2,
        status=DocumentStatus.PENDING,
    )
    session.add(document)
    session.flush()

    count = ingest_document(session, document.id, tenant.id)
    return tenant, policy, document, count


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
    results = retrieve(
        session,
        tenant_id=tenant.id,
        query_vector=fake_embed_one(question),
        policy_ids=[policy.id],
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
    chunks = retrieve(
        session,
        tenant_id=tenant.id,
        query_vector=fake_embed_one(question),
        policy_ids=[policy.id],
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
    chunks = retrieve(
        session,
        tenant_id=tenant.id,
        query_vector=fake_embed_one("cosmetic surgery"),
        policy_ids=[policy.id],
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

    results = retrieve(
        session,
        tenant_id=other.id,
        query_vector=fake_embed_one("waiting period pre-existing disease"),
        top_k=20,
    )
    assert results == []
