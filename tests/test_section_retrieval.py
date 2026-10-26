"""The section-path retrieval arm.

Built to address phase 2's sharpest failure: a term's *definition* outranking the
*clause* that governs it, because the two passages share almost all their body
vocabulary. The section path is the signal that separates them.

It did not work out -- see docs/experiments.md. These tests cover the mechanism, which
is correct and still available behind a flag, rather than asserting it improves ranking.
"""

from __future__ import annotations

import uuid

import pytest

from rag.retrieval import section
from rag.retrieval.search import search
from rag.retrieval.types import RetrievalMode, RetrievedBy
from tests.conftest import fake_embed_one

pytestmark = pytest.mark.db


def test_section_arm_finds_a_chunk_by_its_section_name(ingested, session):
    """"waiting period" appears in the section path of the clause that answers it."""
    tenant, policy, _, _ = ingested
    results = section.retrieve(
        session, tenant.id, "waiting period", policy_ids=[policy.id]
    )
    assert results
    assert all("waiting period" in r.section_path.lower() for r in results)


def test_section_arm_ignores_body_text(ingested, session):
    """It ranks on the address, not the passage -- that is the whole point of the arm.

    "thirty six months" is in the body of the waiting-period clause and in no section
    path, so this arm must return nothing for it.
    """
    tenant, policy, _, _ = ingested
    assert section.retrieve(
        session, tenant.id, "thirty six months", policy_ids=[policy.id]
    ) == []


def test_section_arm_separates_a_definition_from_its_clause(ingested, session):
    """The failure the arm was built for, in isolation.

    Both passages are about pre-existing disease. Only one is under a section path
    naming a waiting period.
    """
    tenant, policy, _, _ = ingested
    results = section.retrieve(
        session, tenant.id, "waiting period pre-existing disease", policy_ids=[policy.id]
    )
    assert results
    top = results[0]
    assert "waiting period" in top.section_path.lower()
    assert "definitions" not in top.section_path.lower()


def test_section_arm_leaves_the_cosine_score_unset(ingested, session):
    tenant, policy, _, _ = ingested
    results = section.retrieve(session, tenant.id, "exclusions", policy_ids=[policy.id])
    assert results
    assert all(r.semantic_score is None for r in results)
    assert all(r.retrieved_by == RetrievedBy.SECTION for r in results)


def test_section_arm_returns_nothing_for_an_unusable_query(ingested, session):
    tenant, policy, _, _ = ingested
    assert section.retrieve(session, tenant.id, "?!", policy_ids=[policy.id]) == []


def test_section_arm_is_tenant_scoped(ingested, session):
    from db.models import Tenant

    other = Tenant(slug=f"sec-{uuid.uuid4().hex[:8]}", name="Other")
    session.add(other)
    session.flush()
    assert section.retrieve(session, other.id, "waiting period") == []


def test_hybrid_section_mode_includes_section_arm_results(ingested, session):
    tenant, policy, _, _ = ingested
    question = "waiting period for pre-existing disease"

    section_only = section.retrieve(
        session, tenant.id, question, policy_ids=[policy.id]
    )
    fused = search(
        session,
        tenant.id,
        question,
        policy_ids=[policy.id],
        mode=RetrievalMode.HYBRID_SECTION,
        query_vector=fake_embed_one(question),
        top_k=10,
    )
    assert {c.chunk_id for c in section_only} <= {c.chunk_id for c in fused}


def test_every_hybrid_section_result_carries_a_confidence_score(ingested, session):
    tenant, policy, _, _ = ingested
    question = "exclusions"
    results = search(
        session,
        tenant.id,
        question,
        policy_ids=[policy.id],
        mode=RetrievalMode.HYBRID_SECTION,
        query_vector=fake_embed_one(question),
    )
    assert results
    assert all(r.semantic_score is not None for r in results)
