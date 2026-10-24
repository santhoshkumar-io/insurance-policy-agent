"""The one retrieval entry point.

The API, the CLI and the evaluation harness all call this, so an experiment measures the
same code path that serves a request. A harness with its own retrieval logic measures the
harness.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from rag.config import get_settings
from rag.retrieval import bm25, keyword, vector
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.types import RetrievalMode, RetrievedChunk


def search(
    session: Session,
    tenant_id: uuid.UUID,
    question: str,
    *,
    policy_ids: list[uuid.UUID] | None = None,
    top_k: int | None = None,
    mode: RetrievalMode | str | None = None,
    rerank: bool | None = None,
    query_vector: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve evidence for ``question``, scoped to the tenant and selected policies.

    ``query_vector`` may be supplied to reuse an embedding across configurations -- the
    experiment matrix runs the same question many times and there is no reason to pay for
    the same embedding each pass.
    """
    settings = get_settings()
    mode = RetrievalMode(mode or settings.retrieval_mode)
    top_k = top_k or settings.retrieval_top_k
    rerank = settings.rerank_enabled if rerank is None else rerank
    candidates = settings.retrieval_candidates

    # Each arm must contribute more than top_k for fusion or reranking to have anything
    # to reorder; a single arm can stop at top_k unless a reranker follows it.
    fan_out = candidates if (mode != RetrievalMode.VECTOR or rerank) else top_k

    vector_hits: list[RetrievedChunk] = []
    if mode.uses_vector:
        if query_vector is None:
            from rag.embeddings.openai_embeddings import embed_query

            query_vector = embed_query(question)
        vector_hits = vector.retrieve(
            session, tenant_id, query_vector, policy_ids=policy_ids, top_k=fan_out
        )

    lexical_hits: list[RetrievedChunk] = []
    if mode in {RetrievalMode.KEYWORD, RetrievalMode.HYBRID}:
        lexical_hits = keyword.retrieve(
            session, tenant_id, question, policy_ids=policy_ids, top_k=fan_out
        )
    elif mode in {RetrievalMode.BM25, RetrievalMode.HYBRID_BM25}:
        lexical_hits = bm25.retrieve(
            session, tenant_id, question, policy_ids=policy_ids, top_k=fan_out
        )

    if mode == RetrievalMode.VECTOR:
        results = vector_hits[: fan_out if rerank else top_k]
    elif mode in {RetrievalMode.KEYWORD, RetrievalMode.BM25}:
        results = lexical_hits[: fan_out if rerank else top_k]
    else:
        results = reciprocal_rank_fusion(
            [vector_hits, lexical_hits],
            k=settings.rrf_k,
            top_k=fan_out if rerank else top_k,
        )

    if rerank and results:
        from rag.retrieval.rerank import rerank as apply_rerank

        results = apply_rerank(question, results, top_k=top_k)

    _fill_semantic_scores(session, tenant_id, question, results, query_vector)
    return results


def _fill_semantic_scores(
    session: Session,
    tenant_id: uuid.UUID,
    question: str,
    results: list[RetrievedChunk],
    query_vector: list[float] | None,
) -> None:
    """Give every result a cosine score, whichever arm found it.

    Without this, a lexical-only result reaches abstention with no confidence signal and
    would be treated as zero -- which is how a working hybrid turns into a system that
    refuses to answer. See docs/decisions.md on the score-scale trap.
    """
    missing = [c.chunk_id for c in results if c.semantic_score is None]
    if not missing:
        return

    if query_vector is None:
        # Keyword-only and BM25-only modes skip embedding the query, but abstention
        # still needs a comparable confidence score, so pay for it here.
        from rag.embeddings.openai_embeddings import embed_query

        query_vector = embed_query(question)

    scores = vector.semantic_scores_for(session, tenant_id, query_vector, missing)
    for chunk in results:
        if chunk.semantic_score is None:
            chunk.semantic_score = scores.get(chunk.chunk_id, 0.0)
