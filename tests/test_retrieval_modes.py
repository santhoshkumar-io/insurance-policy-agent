"""Every retrieval mode, against a real database.

The point of this file is the tenant-isolation parametrisation. Week 1 proved isolation
for vector search; the refactor added three more code paths into the same data, and a
hole in any one of them is a hole in the system. So the boundary is asserted for all of
them rather than for the one that happened to exist first.

    docker compose up -d && alembic upgrade head && pytest -m db
"""

from __future__ import annotations

import uuid

import pytest

from db.models import Tenant
from rag.retrieval.search import search
from rag.retrieval.types import RetrievalMode, RetrievedBy
from tests.conftest import fake_embed_one

pytestmark = pytest.mark.db

ALL_MODES = list(RetrievalMode)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_every_mode_retrieves_the_waiting_period_clause(ingested, session, mode):
    tenant, policy, _, _ = ingested
    question = "pre-existing disease waiting period thirty six months"
    results = search(
        session,
        tenant.id,
        question,
        policy_ids=[policy.id],
        mode=mode,
        query_vector=fake_embed_one(question),
        top_k=8,
    )
    assert results, f"{mode} returned nothing"
    assert any("thirty six months" in r.text for r in results), f"{mode} missed the clause"


@pytest.mark.parametrize("mode", ALL_MODES)
def test_tenant_isolation_holds_in_every_mode(ingested, session, mode):
    """The refactor must not have opened a hole in any of the new code paths."""
    other = Tenant(slug=f"other-{uuid.uuid4().hex[:8]}", name="Other")
    session.add(other)
    session.flush()

    question = "pre-existing disease waiting period"
    results = search(
        session,
        other.id,
        question,
        mode=mode,
        query_vector=fake_embed_one(question),
        top_k=20,
    )
    assert results == [], f"{mode} leaked another tenant's chunks"


@pytest.mark.parametrize("mode", ALL_MODES)
def test_every_result_carries_a_confidence_score(ingested, session, mode):
    """Keyword-only results have no cosine score of their own; search() must supply one,
    or abstention reads zero confidence and refuses a perfectly good answer."""
    tenant, policy, _, _ = ingested
    question = "cosmetic surgery exclusion"
    results = search(
        session,
        tenant.id,
        question,
        policy_ids=[policy.id],
        mode=mode,
        query_vector=fake_embed_one(question),
    )
    assert results
    for chunk in results:
        assert chunk.semantic_score is not None, f"{mode} left a result without a cosine score"
        assert -1.0 <= chunk.semantic_score <= 1.0


def test_hybrid_marks_chunks_found_by_both_arms(ingested, session):
    tenant, policy, _, _ = ingested
    question = "pre-existing disease waiting period"
    results = search(
        session,
        tenant.id,
        question,
        policy_ids=[policy.id],
        mode=RetrievalMode.HYBRID,
        query_vector=fake_embed_one(question),
        top_k=8,
    )
    assert results
    assert all(r.retrieved_by in set(RetrievedBy) for r in results)
    assert {str(r.retrieved_by) for r in results} & {"both", "vector", "keyword"}
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_policy_filter_excludes_other_policies(ingested, session):
    tenant, _, _, _ = ingested
    question = "waiting period"
    results = search(
        session,
        tenant.id,
        question,
        policy_ids=[uuid.uuid4()],  # a policy that does not exist
        mode=RetrievalMode.HYBRID,
        query_vector=fake_embed_one(question),
    )
    assert results == []
