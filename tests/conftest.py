"""Test fixtures.

The policy PDFs are generated rather than vendored (see ``scripts/sample_policy.py``):
a fixture whose exact layout the tests know is the only way to assert that a heading
became a section path and that page 2 really is page 2.
"""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from pathlib import Path

import pytest

from scripts.sample_policy import (
    build_flat_policy,
    build_injected_policy,
    build_lic_shaped_policy,
    build_structured_policy,
)

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")


@pytest.fixture(scope="session")
def structured_policy_pdf(tmp_path_factory) -> Path:
    """A policy with clear headings, numbered clauses and a benefits table."""
    return build_structured_policy(tmp_path_factory.mktemp("pdfs") / "structured_policy.pdf")


@pytest.fixture(scope="session")
def flat_policy_pdf(tmp_path_factory) -> Path:
    """A policy with no structure at all -- must trigger the windowed fallback."""
    return build_flat_policy(tmp_path_factory.mktemp("pdfs") / "flat_policy.pdf")


@pytest.fixture(scope="session")
def lic_shaped_pdf(tmp_path_factory) -> Path:
    """A document with the structural traps the real LIC wordings exposed."""
    return build_lic_shaped_policy(tmp_path_factory.mktemp("pdfs") / "lic_shaped.pdf")


@pytest.fixture(scope="session")
def injected_policy_pdf(tmp_path_factory) -> Path:
    """Red-team fixture: a policy carrying instructions aimed at the model."""
    return build_injected_policy(tmp_path_factory.mktemp("pdfs") / "injected_policy.pdf")


# --------------------------------------------------------------------------------------
# Database-backed fixtures, shared by every `db`-marked module.
# --------------------------------------------------------------------------------------

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
    """Replace the embedding call in the ingestion path. No key, no network."""
    from rag.config import get_settings
    from rag.embeddings.openai_embeddings import EmbeddingResult

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
    """A fresh tenant with the fixture policy ingested.

    Each call makes its own tenant, so tests cannot see one another's corpus -- which
    also means the isolation assertions are meaningful rather than incidental.
    """
    from db.models import Document, DocumentStatus, Policy, Tenant
    from rag.ingestion.pipeline import ingest_document

    tenant = Tenant(slug=f"t-{uuid.uuid4().hex[:8]}", name="Test Tenant")
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
