"""Retrieval and abstention metrics, against hand-computed values."""

from __future__ import annotations

import math

from rag.evaluation.dataset import EvalQuestion
from rag.evaluation.metrics import (
    QuestionResult,
    aggregate,
    hits,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    section_hit,
)
from tests.test_citation_validation import make_chunk


def question(pages=(2,), answerable=True, section=None) -> EvalQuestion:
    return EvalQuestion(
        id="q1",
        question="waiting period?",
        policy="acme",
        answerable=answerable,
        expected_pages=list(pages),
        expected_section=section,
    )


def chunk_on(page_from: int, page_to: int | None = None, section: str = ""):
    chunk = make_chunk()
    chunk.page_from = page_from
    chunk.page_to = page_to if page_to is not None else page_from
    chunk.section_path = section
    return chunk


def test_page_level_matching_is_an_intersection_not_equality():
    """A chunk spanning pages 1-3 contains page 2, so it is a hit.

    Labels are page-based so they survive re-chunking; a chunk that straddles the
    labelled page must count, or every chunking change would move the metric.
    """
    q = question(pages=(2,))
    assert hits(q, [chunk_on(1, 3)]) == [True]
    assert hits(q, [chunk_on(3, 5)]) == [False]
    assert hits(q, [chunk_on(2)]) == [True]


def test_recall_at_k_respects_the_cut_off():
    q = question(pages=(2,))
    ranked = [chunk_on(7), chunk_on(8), chunk_on(2)]
    assert recall_at_k(q, ranked, k=2) == 0.0
    assert recall_at_k(q, ranked, k=3) == 1.0


def test_reciprocal_rank_is_one_over_the_first_hit():
    q = question(pages=(2,))
    assert reciprocal_rank(q, [chunk_on(2)]) == 1.0
    assert reciprocal_rank(q, [chunk_on(9), chunk_on(2)]) == 0.5
    assert reciprocal_rank(q, [chunk_on(9), chunk_on(8), chunk_on(2)]) == 1 / 3
    assert reciprocal_rank(q, [chunk_on(9)]) == 0.0


def test_ndcg_matches_the_hand_computed_value():
    q = question(pages=(2,))
    # One relevant item at rank 2: DCG = 1/log2(3), ideal DCG = 1/log2(2) = 1.
    assert ndcg_at_k(q, [chunk_on(9), chunk_on(2)], k=5) == 1 / math.log2(3)
    assert ndcg_at_k(q, [chunk_on(2)], k=5) == 1.0
    assert ndcg_at_k(q, [chunk_on(9)], k=5) == 0.0


def test_section_hit_is_stricter_than_the_page_hit():
    """Right page, wrong clause -- the page metric passes and this one must not."""
    q = question(pages=(2,), section="4. Waiting Periods > 4.3 Pre-existing disease")
    right_page_wrong_section = [chunk_on(2, section="4. Waiting Periods > 4.1 Initial")]
    assert recall_at_k(q, right_page_wrong_section, k=5) == 1.0
    assert section_hit(q, right_page_wrong_section, k=5) is False

    correct = [chunk_on(2, section="4. Waiting Periods > 4.3 Pre-existing disease")]
    assert section_hit(q, correct, k=5) is True


def test_unlabelled_section_never_counts_as_a_hit():
    q = question(pages=(2,), section=None)
    assert section_hit(q, [chunk_on(2, section="anything")], k=5) is False


def test_retrieval_quality_averages_over_answerable_questions_only():
    """An unanswerable question has no evidence to find; scoring it would dilute recall."""
    results = [
        QuestionResult("q1", answerable=True, recall=1.0, rr=1.0, ndcg=1.0, section=True),
        QuestionResult("q2", answerable=True, recall=0.0, rr=0.0, ndcg=0.0),
        QuestionResult("q3", answerable=False, recall=0.0, rr=0.0, ndcg=0.0),
    ]
    metrics = aggregate("test", k=5, results=results)
    assert metrics.recall_at_k == 0.5  # not 1/3
    assert metrics.answerable == 2
    assert metrics.unanswerable == 1


def test_both_abstention_error_directions_are_reported():
    results = [
        # answerable but abstained -> a false abstention
        QuestionResult("q1", answerable=True, abstained=True, tokens=100),
        QuestionResult("q2", answerable=True, abstained=False, tokens=100),
        # unanswerable but answered -> a false answer
        QuestionResult("q3", answerable=False, abstained=False, tokens=100),
        QuestionResult("q4", answerable=False, abstained=True, tokens=100),
    ]
    metrics = aggregate("test", k=5, results=results)
    assert metrics.false_abstention_rate == 0.5
    assert metrics.false_answer_rate == 0.5
    assert metrics.tokens_per_100q == 10000


def test_retrieval_only_runs_leave_the_generation_columns_empty():
    """An em dash in the table is honest; a zero would read as a measured result."""
    results = [QuestionResult("q1", answerable=True, recall=1.0)]
    metrics = aggregate("test", k=5, results=results)
    assert metrics.false_answer_rate is None
    assert metrics.tokens_per_100q is None
    row = metrics.as_row()
    assert row["false answer"] == "—"
    assert row["tokens/100q"] == "—"
