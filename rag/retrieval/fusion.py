"""Reciprocal Rank Fusion.

    score(d) = Σ_arms 1 / (k + rank_arm(d))

Chosen over weighted score fusion because the two arms produce incomparable scales --
cosine similarity in [-1, 1] and ts_rank_cd unbounded and corpus-dependent. RRF throws
the scores away and keeps only the ranks, so there is nothing to normalise and no
weighting to tune on a benchmark this small.

A document found by both arms accumulates two reciprocal terms, which is the property
that makes the hybrid worth having: agreement between a semantic and a lexical match is
genuine evidence, and RRF rewards it without anyone hand-tuning a weight.
"""

from __future__ import annotations

from rag.retrieval.types import RetrievedBy, RetrievedChunk

DEFAULT_K = 60


def reciprocal_rank_fusion(
    arms: list[list[RetrievedChunk]], k: int = DEFAULT_K, top_k: int | None = None
) -> list[RetrievedChunk]:
    """Fuse ranked lists. Each arm must already be ordered best-first."""
    scores: dict = {}
    best: dict = {}
    sources: dict = {}

    for arm in arms:
        for position, chunk in enumerate(arm, start=1):
            key = chunk.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position)
            sources.setdefault(key, set()).add(chunk.retrieved_by)
            # Keep whichever copy carries a semantic score, so the confidence signal
            # survives fusion even when the keyword arm ranked the chunk higher.
            existing = best.get(key)
            if existing is None or (
                existing.semantic_score is None and chunk.semantic_score is not None
            ):
                best[key] = chunk

    fused: list[RetrievedChunk] = []
    for key, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        chunk = best[key]
        found = sources[key]
        chunk.score = score
        chunk.retrieved_by = (
            RetrievedBy.BOTH if len(found) > 1 else next(iter(found))
        )
        fused.append(chunk)

    if top_k is not None:
        fused = fused[:top_k]
    for position, chunk in enumerate(fused, start=1):
        chunk.rank = position
    return fused
