"""Keyword retrieval using PostgreSQL full-text search.

This is ``ts_rank_cd``, **not BM25**. Stock PostgreSQL does not provide BM25: there is no
document-length normalisation of the BM25 form and no k1/b parameters. The brief's stack
table says BM25; this project reports what it actually implements, and runs real BM25 as
a measured experiment instead of claiming it (plan.md §2.3).

Why a keyword arm exists at all: embeddings are good at paraphrase and bad at exact
tokens. An agent searching for a clause name, a defined term or a UIN wants the literal
string, and that is where cosine similarity is weakest.
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session

from db.models import Chunk
from rag.config import get_settings
from rag.retrieval.types import RetrievedBy, RetrievedChunk
from rag.retrieval.vector import base_statement, build_chunk

# Keep alphanumerics and the separators that appear inside policy terms (UIN codes,
# clause numbers); everything else would only confuse the tsquery parser.
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9./-]*")


def to_tsquery_text(question: str) -> str:
    """Turn a natural-language question into an OR-ed tsquery string.

    OR rather than AND on purpose: this arm supplies *candidates* for fusion, so recall
    matters more than precision here. An AND query over a full sentence returns nothing
    for almost every real question.
    """
    tokens = [t for t in _TOKEN.findall(question) if len(t) > 1]
    return " | ".join(tokens)


def retrieve(
    session: Session,
    tenant_id: uuid.UUID,
    question: str,
    policy_ids: list[uuid.UUID] | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or get_settings().retrieval_top_k

    query_text = to_tsquery_text(question)
    if not query_text:
        return []

    tsquery = func.to_tsquery("english", query_text)
    rank = func.ts_rank_cd(Chunk.text_tsv, tsquery).label("rank_score")

    stmt = (
        base_statement(tenant_id, policy_ids)
        .add_columns(rank)
        .where(Chunk.text_tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(top_k)
    )

    results: list[RetrievedChunk] = []
    for position, (chunk, filename, product_name, score) in enumerate(
        session.execute(stmt).all(), start=1
    ):
        results.append(
            build_chunk(
                chunk,
                filename,
                product_name,
                score=float(score),
                # Deliberately left None: ts_rank_cd is not comparable to cosine
                # similarity. search() fills it in for the chunks it keeps.
                semantic_score=None,
                rank=position,
                retrieved_by=RetrievedBy.KEYWORD,
            )
        )
    return results
