"""In-memory BM25, for comparison only.

PostgreSQL cannot do BM25 (see rag/retrieval/keyword.py). Rather than assert that BM25
would do better than ts_rank_cd, this runs it over the scoped corpus and lets the
benchmark decide.

It is honest about its own limits: the whole candidate set is loaded and indexed per
query, which is fine for a benchmark corpus and would not be fine in production. That is
exactly why it is an experiment arm and not a shipping mode -- if it wins decisively, the
answer is the ParadeDB pg_search extension, not this.
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy.orm import Session

from rag.config import get_settings
from rag.retrieval.types import RetrievedBy, RetrievedChunk
from rag.retrieval.vector import base_statement, build_chunk

_TOKEN = re.compile(r"[a-z0-9][a-z0-9./-]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def retrieve(
    session: Session,
    tenant_id: uuid.UUID,
    question: str,
    policy_ids: list[uuid.UUID] | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("BM25 comparison needs `rank_bm25`.") from exc

    top_k = top_k or get_settings().retrieval_top_k

    rows = session.execute(base_statement(tenant_id, policy_ids)).all()
    if not rows:
        return []

    corpus = [tokenize(chunk.text) for chunk, _, _ in rows]
    scores = BM25Okapi(corpus).get_scores(tokenize(question))

    ranked = sorted(
        zip(rows, scores, strict=True), key=lambda pair: float(pair[1]), reverse=True
    )

    results: list[RetrievedChunk] = []
    for position, ((chunk, filename, product_name), score) in enumerate(
        ranked[:top_k], start=1
    ):
        if float(score) <= 0.0:
            break  # no term overlap at all; padding the list would only add noise
        results.append(
            build_chunk(
                chunk,
                filename,
                product_name,
                score=float(score),
                semantic_score=None,
                rank=position,
                retrieved_by=RetrievedBy.KEYWORD,
            )
        )
    return results
