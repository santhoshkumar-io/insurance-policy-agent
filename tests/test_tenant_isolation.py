"""Tenant isolation, tested from day 1 even though auth is still stubbed.

The boundary is a schema and query property, not a feature that arrives in week 5
(plan.md §2.5). These tests need a live PostgreSQL with pgvector:

    docker compose up -d && alembic upgrade head && pytest -m db
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from db.models import Chunk, Document, Policy, Tenant
from db.session import scoped
from rag.config import get_settings
from rag.retrieval.vector import retrieve

pytestmark = pytest.mark.db


@pytest.fixture
def session():
    from db.session import get_session_factory

    session = get_session_factory()()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def _make_tenant_with_chunk(session, slug: str, text: str, vector: list[float]):
    settings = get_settings()
    tenant = Tenant(slug=slug, name=slug)
    session.add(tenant)
    session.flush()

    policy = Policy(tenant_id=tenant.id, product_name=f"{slug} product")
    session.add(policy)
    session.flush()

    document = Document(
        tenant_id=tenant.id,
        policy_id=policy.id,
        filename=f"{slug}.pdf",
        storage_path=f"/tmp/{slug}.pdf",
        checksum=uuid.uuid4().hex + uuid.uuid4().hex[:0],
        status="ready",
        page_count=1,
    )
    session.add(document)
    session.flush()

    chunk = Chunk(
        tenant_id=tenant.id,
        document_id=document.id,
        policy_id=policy.id,
        ordinal=0,
        page_from=1,
        page_to=1,
        section_path="1. Coverage",
        chunk_type="prose",
        text=text,
        token_count=10,
        embedding=vector,
        embedding_model=settings.openai_embedding_model,
        embedding_dim=settings.openai_embedding_dim,
    )
    session.add(chunk)
    session.flush()
    return tenant, policy, chunk


def test_retrieval_never_crosses_the_tenant_boundary(session):
    dim = get_settings().openai_embedding_dim
    vector = [0.1] * dim

    a_slug, b_slug = f"a-{uuid.uuid4().hex[:8]}", f"b-{uuid.uuid4().hex[:8]}"
    tenant_a, _, chunk_a = _make_tenant_with_chunk(session, a_slug, "TENANT A SECRET", vector)
    tenant_b, _, chunk_b = _make_tenant_with_chunk(session, b_slug, "TENANT B SECRET", vector)

    # Identical vectors: without scoping, both would rank equally and both come back.
    results_a = retrieve(session, tenant_id=tenant_a.id, query_vector=vector, top_k=50)
    returned = {r.chunk_id for r in results_a}

    assert chunk_a.id in returned
    assert chunk_b.id not in returned
    assert all("TENANT B" not in r.text for r in results_a)


def test_scoped_refuses_a_model_without_a_tenant_column():
    with pytest.raises(TypeError):
        scoped(select(Tenant), Tenant, uuid.uuid4())


def test_scoped_filters_documents(session):
    dim = get_settings().openai_embedding_dim
    tenant_a, _, _ = _make_tenant_with_chunk(
        session, f"a-{uuid.uuid4().hex[:8]}", "A", [0.2] * dim
    )
    tenant_b, _, _ = _make_tenant_with_chunk(
        session, f"b-{uuid.uuid4().hex[:8]}", "B", [0.2] * dim
    )

    docs_a = list(session.scalars(scoped(select(Document), Document, tenant_a.id)))
    assert docs_a
    assert all(d.tenant_id == tenant_a.id for d in docs_a)
