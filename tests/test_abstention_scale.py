"""Regression guard for the score-scale trap.

The abstention threshold was tuned for cosine similarity. An RRF score is ~0.016 and a
cross-encoder logit is unbounded and often negative, so a threshold applied to the
ranking score would reject every answer the moment hybrid retrieval was switched on --
and every retrieval experiment would then be measuring abstention, not retrieval.

RetrievedChunk.confidence exists so the threshold means the same thing under every
ranking mode. These tests fail if that separation is ever collapsed.
"""

from __future__ import annotations

from rag.config import get_settings
from rag.retrieval.types import RetrievedChunk
from tests.test_citation_validation import make_chunk
from tests.test_generation import StubModel, answerable


def good_evidence(ranking_score: float) -> RetrievedChunk:
    """Strong semantic match, but ranked by a scale of its own."""
    return make_chunk(score=ranking_score, semantic_score=0.61)


def test_rrf_scale_ranking_does_not_trigger_abstention():
    from rag.generation.answer import generate_answer

    chunk = good_evidence(1 / 61)  # a typical top RRF score
    assert chunk.score < get_settings().abstain_semantic_threshold

    answer = generate_answer("waiting period?", [chunk], model=StubModel([answerable(chunk)]))
    assert answer.answerable, "hybrid ranking must not be read as low confidence"


def test_negative_cross_encoder_logit_does_not_trigger_abstention():
    from rag.generation.answer import generate_answer

    chunk = good_evidence(-4.2)
    answer = generate_answer("waiting period?", [chunk], model=StubModel([answerable(chunk)]))
    assert answer.answerable


def test_weak_evidence_still_abstains_whatever_the_ranking_score():
    """The guard must not become a blanket exemption -- low confidence still abstains."""
    from rag.generation.answer import generate_answer

    threshold = get_settings().abstain_semantic_threshold
    chunk = make_chunk(score=0.99, semantic_score=threshold - 0.05)

    answer = generate_answer("anything?", [chunk], model=StubModel([answerable(chunk)]))
    assert not answer.answerable
    assert "below the threshold" in answer.abstained_reason


def test_abstention_decision_is_identical_across_ranking_scales():
    from rag.generation.answer import generate_answer

    decisions = []
    for ranking_score in (0.61, 1 / 61, -4.2, 12.7):
        chunk = good_evidence(ranking_score)
        answer = generate_answer("q?", [chunk], model=StubModel([answerable(chunk)]))
        decisions.append(answer.answerable)
    assert len(set(decisions)) == 1, "the same evidence decided differently per scale"


def test_confidence_falls_back_to_score_when_no_semantic_score_exists():
    chunk = RetrievedChunk(
        chunk_id=make_chunk().chunk_id,
        text="t",
        page_from=1,
        page_to=1,
        section_path="",
        chunk_type="prose",
        policy_id=make_chunk().policy_id,
        document_id=make_chunk().document_id,
        document_name="d",
        policy_name="p",
        score=0.5,
    )
    assert chunk.confidence == 0.5
