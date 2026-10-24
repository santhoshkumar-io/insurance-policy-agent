"""Reciprocal Rank Fusion."""

from __future__ import annotations

from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.types import RetrievedBy
from tests.test_citation_validation import make_chunk


def arm(chunks, by: RetrievedBy):
    for position, chunk in enumerate(chunks, start=1):
        chunk.rank = position
        chunk.retrieved_by = by
    return chunks


def test_agreement_between_arms_outranks_a_single_arm_hit():
    """The property that makes a hybrid worth having.

    A chunk both arms found at rank 2 should beat one only the vector arm found at
    rank 1 -- semantic and lexical agreement is real evidence, and RRF rewards it
    without anyone hand-tuning a weight.
    """
    shared_v = make_chunk(text="agreed")
    shared_k = make_chunk(text="agreed")
    shared_k.chunk_id = shared_v.chunk_id  # same chunk, found twice
    solo = make_chunk(text="vector only")

    fused = reciprocal_rank_fusion(
        [
            arm([solo, shared_v], RetrievedBy.VECTOR),
            arm([make_chunk(), shared_k], RetrievedBy.KEYWORD),
        ]
    )

    assert fused[0].chunk_id == shared_v.chunk_id
    assert fused[0].retrieved_by == RetrievedBy.BOTH


def test_single_arm_results_keep_their_own_provenance():
    v = make_chunk(text="v")
    k = make_chunk(text="k")
    fused = reciprocal_rank_fusion([arm([v], RetrievedBy.VECTOR), arm([k], RetrievedBy.KEYWORD)])
    by_id = {c.chunk_id: c.retrieved_by for c in fused}
    assert by_id[v.chunk_id] == RetrievedBy.VECTOR
    assert by_id[k.chunk_id] == RetrievedBy.KEYWORD


def test_scores_follow_the_rrf_formula_and_ranks_are_renumbered():
    a, b = make_chunk(text="a"), make_chunk(text="b")
    fused = reciprocal_rank_fusion([arm([a, b], RetrievedBy.VECTOR)], k=60)
    assert fused[0].score == 1 / 61
    assert fused[1].score == 1 / 62
    assert [c.rank for c in fused] == [1, 2]


def test_semantic_score_survives_fusion_even_when_the_keyword_copy_ranks_higher():
    """Abstention reads the cosine score; fusion must not discard the copy that has it."""
    with_score = make_chunk(text="same", semantic_score=0.61)
    without = make_chunk(text="same")
    without.chunk_id = with_score.chunk_id
    without.semantic_score = None

    fused = reciprocal_rank_fusion(
        [arm([make_chunk(), with_score], RetrievedBy.VECTOR), arm([without], RetrievedBy.KEYWORD)]
    )
    merged = next(c for c in fused if c.chunk_id == with_score.chunk_id)
    assert merged.semantic_score == 0.61


def test_top_k_truncates_after_fusion_not_before():
    chunks = [make_chunk(text=str(i)) for i in range(10)]
    fused = reciprocal_rank_fusion([arm(chunks, RetrievedBy.VECTOR)], top_k=3)
    assert len(fused) == 3
    assert [c.rank for c in fused] == [1, 2, 3]


def test_empty_arms_fuse_to_nothing():
    assert reciprocal_rank_fusion([[], []]) == []
