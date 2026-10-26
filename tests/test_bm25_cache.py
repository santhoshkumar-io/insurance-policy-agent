"""The cached BM25 index.

The arm was previously rebuilt from a full table scan on every query, which is why the
retrieval default stayed on vector search even after hybrid measured better. Caching it is
what makes the winning configuration deployable.

The first test is the one that matters most: a cache keyed without the tenant would serve
one tenant's index to another, and that is a security failure rather than a slow query.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from db.models import Chunk, Document, DocumentStatus, Policy, Tenant
from rag.retrieval import bm25

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
def clear_cache():
    """Each test starts cold; a shared index would make these pass by accident."""
    bm25.CACHE.clear()
    yield
    bm25.CACHE.clear()


def ingest_for(session, slug: str, pdf) -> tuple:
    from rag.ingestion.pipeline import ingest_document

    tenant = Tenant(slug=f"{slug}-{uuid.uuid4().hex[:8]}", name=slug)
    session.add(tenant)
    session.flush()
    policy = Policy(tenant_id=tenant.id, insurer="Acme", product_name=f"{slug} product")
    session.add(policy)
    session.flush()
    document = Document(
        tenant_id=tenant.id,
        policy_id=policy.id,
        filename=f"{slug}.pdf",
        storage_path=str(pdf),
        checksum=uuid.uuid4().hex * 2,
        status=DocumentStatus.PENDING,
    )
    session.add(document)
    session.flush()
    ingest_document(session, document.id, tenant.id)
    return tenant, policy, document


def test_a_cached_index_is_never_served_across_tenants(session, stub_openai, structured_policy_pdf):
    """The security-relevant case. Two tenants, same corpus, same policy-less scope."""
    tenant_a, _, _ = ingest_for(session, "a", structured_policy_pdf)
    tenant_b, _, _ = ingest_for(session, "b", structured_policy_pdf)

    a_results = bm25.retrieve(session, tenant_a.id, "prorated room rent", top_k=20)
    b_results = bm25.retrieve(session, tenant_b.id, "prorated room rent", top_k=20)

    assert a_results and b_results
    a_ids = {c.chunk_id for c in a_results}
    b_ids = {c.chunk_id for c in b_results}
    assert not (a_ids & b_ids), "a cached index leaked between tenants"

    a_chunk_ids = set(
        session.scalars(select(Chunk.id).where(Chunk.tenant_id == tenant_a.id))
    )
    assert a_ids <= a_chunk_ids


def test_an_empty_tenant_gets_nothing_even_when_another_tenant_is_cached(
    session, stub_openai, structured_policy_pdf
):
    tenant_a, _, _ = ingest_for(session, "a", structured_policy_pdf)
    bm25.retrieve(session, tenant_a.id, "prorated", top_k=10)  # warm the cache

    empty = Tenant(slug=f"empty-{uuid.uuid4().hex[:8]}", name="empty")
    session.add(empty)
    session.flush()
    assert bm25.retrieve(session, empty.id, "prorated", top_k=10) == []


def test_the_index_is_reused_across_queries(session, stub_openai, structured_policy_pdf):
    tenant, policy, _ = ingest_for(session, "reuse", structured_policy_pdf)

    bm25.retrieve(session, tenant.id, "prorated", policy_ids=[policy.id])
    after_first = bm25.CACHE.hits

    for _ in range(3):
        bm25.retrieve(session, tenant.id, "cosmetic surgery", policy_ids=[policy.id])

    assert bm25.CACHE.hits == after_first + 3
    assert bm25.CACHE.stats()["scopes_cached"] == 1


def test_the_index_rebuilds_when_the_corpus_changes(session, stub_openai, structured_policy_pdf):
    """The fingerprint, not an invalidation hook.

    Ingestion runs in a background task and, deployed, possibly in another process. A
    hook would leave every other worker serving a stale index with no way to notice.
    """
    from rag.ingestion.pipeline import ingest_document

    tenant, _, document = ingest_for(session, "stale", structured_policy_pdf)
    bm25.retrieve(session, tenant.id, "prorated")
    before = bm25.CACHE.rebuilds

    # Re-index: same content, new chunk rows, so the fingerprint must move.
    ingest_document(session, document.id, tenant.id)
    results = bm25.retrieve(session, tenant.id, "prorated")

    assert bm25.CACHE.rebuilds == before + 1, "a stale index was served after re-indexing"
    live_ids = set(session.scalars(select(Chunk.id).where(Chunk.document_id == document.id)))
    assert {c.chunk_id for c in results} <= live_ids, "results pointed at deleted chunks"


def test_a_narrower_policy_scope_is_cached_separately(
    session, stub_openai, structured_policy_pdf
):
    tenant, policy, _ = ingest_for(session, "scopes", structured_policy_pdf)

    bm25.retrieve(session, tenant.id, "prorated")                        # whole tenant
    bm25.retrieve(session, tenant.id, "prorated", policy_ids=[policy.id])  # one policy

    assert bm25.CACHE.stats()["scopes_cached"] == 2


def test_results_are_ranked_and_scored(session, stub_openai, structured_policy_pdf):
    tenant, policy, _ = ingest_for(session, "rank", structured_policy_pdf)
    results = bm25.retrieve(session, tenant.id, "prorated room rent", policy_ids=[policy.id])

    assert results
    assert [c.rank for c in results] == list(range(1, len(results) + 1))
    assert all(c.score > 0 for c in results), "zero-overlap chunks should be dropped"
    scores = [c.score for c in results]
    assert scores == sorted(scores, reverse=True)
    # ts_rank_cd and BM25 are not comparable to cosine; search() fills this in later.
    assert all(c.semantic_score is None for c in results)


def test_a_query_with_no_term_overlap_returns_nothing(
    session, stub_openai, structured_policy_pdf
):
    tenant, policy, _ = ingest_for(session, "nomatch", structured_policy_pdf)
    assert bm25.retrieve(
        session, tenant.id, "zzzzqqq unrelated gibberish", policy_ids=[policy.id]
    ) == []


def test_the_cache_is_bounded(session, stub_openai, structured_policy_pdf, monkeypatch):
    """Unbounded growth would be a slow memory leak in a long-running worker."""
    from rag.config import get_settings

    monkeypatch.setattr(get_settings(), "bm25_cache_size", 2)
    tenant, policy, _ = ingest_for(session, "bound", structured_policy_pdf)

    bm25.retrieve(session, tenant.id, "prorated")
    bm25.retrieve(session, tenant.id, "prorated", policy_ids=[policy.id])
    bm25.retrieve(session, tenant.id, "prorated", policy_ids=[policy.id, uuid.uuid4()])

    assert bm25.CACHE.stats()["scopes_cached"] <= 2


def test_stats_report_the_build_cost(session, stub_openai, structured_policy_pdf):
    """Published rather than asserted to be small -- see the module docstring."""
    tenant, _, _ = ingest_for(session, "stats", structured_policy_pdf)
    bm25.retrieve(session, tenant.id, "prorated")

    entries = bm25.CACHE.stats()["entries"]
    assert entries
    assert entries[0]["chunks"] > 0
    assert entries[0]["tokens"] > 0
    assert entries[0]["build_ms"] >= 0


def test_the_cached_index_scores_identically_to_a_fresh_one(
    session, stub_openai, structured_policy_pdf
):
    """Equivalence with the uncached implementation, bit for bit.

    This test exists because it was needed. A borderline benchmark question flipped
    behaviour after the cache landed, and the cache was the obvious suspect. Proving the
    rankings and scores were identical is what redirected the investigation to the real
    cause -- a mislabelled question -- instead of to a rewrite of working code.

    Caching is only a safe optimisation if it changes nothing but latency.
    """
    from rank_bm25 import BM25Okapi

    from rag.retrieval.vector import base_statement

    tenant, policy, _ = ingest_for(session, "equiv", structured_policy_pdf)
    question = "prorated room rent eligible category"

    # The original approach: read every scoped row, index it, score it, all per query.
    rows = session.execute(base_statement(tenant.id, [policy.id])).all()
    corpus = [bm25.tokenize(chunk.text) for chunk, _, _ in rows]
    fresh_scores = BM25Okapi(corpus).get_scores(bm25.tokenize(question))
    fresh = sorted(
        ((chunk.id, float(s)) for (chunk, _, _), s in zip(rows, fresh_scores, strict=True)),
        key=lambda pair: -pair[1],
    )
    fresh = [(cid, score) for cid, score in fresh if score > 0][:8]

    bm25.CACHE.clear()
    cached = [
        (c.chunk_id, c.score)
        for c in bm25.retrieve(session, tenant.id, question, policy_ids=[policy.id], top_k=8)
    ]

    assert [cid for cid, _ in cached] == [cid for cid, _ in fresh], "ranking changed"
    for (_, cached_score), (_, fresh_score) in zip(cached, fresh, strict=True):
        assert cached_score == pytest.approx(fresh_score), "scores changed"


def test_a_warm_index_gives_the_same_answer_as_a_cold_one(
    session, stub_openai, structured_policy_pdf
):
    """The same query twice must not depend on whether the index was already built."""
    tenant, policy, _ = ingest_for(session, "warm", structured_policy_pdf)
    question = "cosmetic surgery exclusion"

    bm25.CACHE.clear()
    cold = bm25.retrieve(session, tenant.id, question, policy_ids=[policy.id], top_k=8)
    warm = bm25.retrieve(session, tenant.id, question, policy_ids=[policy.id], top_k=8)

    assert [c.chunk_id for c in cold] == [c.chunk_id for c in warm]
    assert [c.score for c in cold] == [c.score for c in warm]
