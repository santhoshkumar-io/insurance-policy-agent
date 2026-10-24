"""The keyword arm — why the hybrid exists.

Embeddings are good at paraphrase and weak on exact tokens. An agent hunting a defined
term, a clause number or a rare word wants the literal string. These tests cover that
behaviour and the query translation around it.
"""

from __future__ import annotations

import uuid

import pytest

from rag.retrieval import keyword
from rag.retrieval.search import search
from rag.retrieval.types import RetrievalMode
from tests.conftest import fake_embed_one

# ---------------------------------------------------------------------------- unit


def test_tsquery_keeps_policy_style_tokens():
    """Clause numbers and UINs are exactly what someone searches literally."""
    query = keyword.to_tsquery_text("What does clause 4.2 of UIN ACM0001 say?")
    assert "4.2" in query
    assert "ACM0001" in query
    assert " | " in query, "candidate generation should OR terms, not AND them"


def test_tsquery_drops_punctuation_that_would_break_the_parser():
    query = keyword.to_tsquery_text("Is 'cosmetic surgery' covered (fully)?")
    assert "&" not in query and "'" not in query and "(" not in query
    assert "cosmetic" in query and "surgery" in query


def test_tsquery_of_an_unusable_question_is_empty():
    assert keyword.to_tsquery_text("?? !") == ""
    assert keyword.to_tsquery_text("a I") == ""


# ------------------------------------------------------------------------------ db

pytestmark_db = pytest.mark.db


@pytest.mark.db
def test_keyword_arm_finds_a_rare_exact_term(ingested, session):
    """'prorated' appears once, in the room-rent clause. Literal match should land it."""
    tenant, policy, _, _ = ingested
    results = keyword.retrieve(session, tenant.id, "prorated", policy_ids=[policy.id])
    assert results, "exact term search returned nothing"
    assert any("prorated" in r.text for r in results)
    assert all(r.score > 0 for r in results)


@pytest.mark.db
def test_keyword_arm_returns_nothing_rather_than_everything_for_an_empty_query(
    ingested, session
):
    tenant, policy, _, _ = ingested
    assert keyword.retrieve(session, tenant.id, "?!", policy_ids=[policy.id]) == []


@pytest.mark.db
def test_keyword_arm_leaves_the_cosine_score_unset(ingested, session):
    """ts_rank_cd is not comparable to cosine similarity, so it must not masquerade as it.
    search() fills the gap in; the arm itself does not invent one."""
    tenant, policy, _, _ = ingested
    results = keyword.retrieve(session, tenant.id, "prorated", policy_ids=[policy.id])
    assert results
    assert all(r.semantic_score is None for r in results)


@pytest.mark.db
def test_hybrid_includes_what_the_keyword_arm_alone_found(ingested, session):
    """A chunk the vector arm ranked outside the cut should still reach the model."""
    tenant, policy, _, _ = ingested
    question = "prorated"

    narrow_vector = search(
        session,
        tenant.id,
        question,
        policy_ids=[policy.id],
        mode=RetrievalMode.VECTOR,
        query_vector=fake_embed_one(question),
        top_k=1,
    )
    keyword_only = keyword.retrieve(session, tenant.id, question, policy_ids=[policy.id])
    hybrid = search(
        session,
        tenant.id,
        question,
        policy_ids=[policy.id],
        mode=RetrievalMode.HYBRID,
        query_vector=fake_embed_one(question),
        top_k=8,
    )

    hybrid_ids = {c.chunk_id for c in hybrid}
    assert {c.chunk_id for c in keyword_only} <= hybrid_ids, "fusion dropped keyword hits"
    assert {c.chunk_id for c in narrow_vector} <= hybrid_ids, "fusion dropped vector hits"


@pytest.mark.db
def test_keyword_arm_is_tenant_scoped(ingested, session):
    from db.models import Tenant

    other = Tenant(slug=f"kw-{uuid.uuid4().hex[:8]}", name="Other")
    session.add(other)
    session.flush()
    assert keyword.retrieve(session, other.id, "prorated") == []
