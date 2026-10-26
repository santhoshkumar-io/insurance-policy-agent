"""The section tie-breaker: a retry of a measured failure, in a different shape.

As a fourth RRF arm, section metadata cost the best configuration 10 points of recall,
because a section path identifies a *region* and the arm flooded fusion with every chunk
in it. These tests pin the contract that makes the tie-breaker a different proposition:
it reorders, it never introduces, and it cannot override a real score difference.
"""

from __future__ import annotations

import uuid

from rag.retrieval.section import apply_tiebreak, matches_section
from rag.retrieval.types import RetrievedChunk


def chunk(score: float, section_path: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        text="clause text",
        page_from=1,
        page_to=1,
        section_path=section_path,
        chunk_type="prose",
        policy_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_name="policy.pdf",
        policy_name="Health Guard",
        score=score,
        semantic_score=0.6,
    )


DEFINITION = "1. Definitions > 1.1 Pre-existing Disease"
CLAUSE = "4. Waiting Periods > 4.3 Pre-existing disease waiting period"


def test_it_fixes_the_failure_it_was_built_for():
    """The phase-2 finding: a term's definition outranking the clause that governs it.

    Both passages share almost all their body vocabulary, so fusion scores them nearly
    equally. The section path is the only thing that separates them.
    """
    results = apply_tiebreak(
        "What is the waiting period for pre-existing diseases?",
        [chunk(0.0320, DEFINITION), chunk(0.0318, CLAUSE)],
    )
    assert results[0].section_path == CLAUSE


def test_it_cannot_override_a_real_score_difference():
    """A chunk fusion ranked clearly higher stays higher, whatever its section says."""
    results = apply_tiebreak(
        "waiting period", [chunk(0.90, DEFINITION), chunk(0.10, CLAUSE)]
    )
    assert results[0].section_path == DEFINITION


def test_the_output_is_always_a_permutation_of_the_input():
    """The contract the arm version broke. Nothing added, nothing dropped."""
    given = [chunk(0.5, DEFINITION), chunk(0.49, CLAUSE), chunk(0.1, "5. Exclusions")]
    results = apply_tiebreak("waiting period exclusions", given)
    assert len(results) == len(given)
    assert {c.chunk_id for c in results} == {c.chunk_id for c in given}


def test_ranks_are_renumbered_after_reordering():
    results = apply_tiebreak(
        "waiting period", [chunk(0.0320, DEFINITION), chunk(0.0318, CLAUSE)]
    )
    assert [c.rank for c in results] == [1, 2]


def test_a_tie_with_no_section_match_keeps_fusion_order():
    """Stable: the tie-breaker is silent when it has nothing to say."""
    first, second = chunk(0.50, "5. Exclusions"), chunk(0.499, "6. Table of Benefits")
    results = apply_tiebreak("grace period", [first, second])
    assert [c.chunk_id for c in results] == [first.chunk_id, second.chunk_id]


def test_tolerance_is_relative_so_it_works_across_score_scales():
    """One tolerance has to serve cosine (~0.6), RRF (~0.03) and logits (unbounded)."""
    # RRF scale: a 0.6% difference is within the band.
    rrf = apply_tiebreak("waiting period", [chunk(0.0320, DEFINITION), chunk(0.0318, CLAUSE)])
    assert rrf[0].section_path == CLAUSE
    # Cosine scale: the same *relative* gap, much larger in absolute terms.
    cosine = apply_tiebreak("waiting period", [chunk(0.640, DEFINITION), chunk(0.636, CLAUSE)])
    assert cosine[0].section_path == CLAUSE


def test_a_single_result_is_returned_untouched():
    one = chunk(0.5, DEFINITION)
    assert apply_tiebreak("anything", [one]) == [one]
    assert apply_tiebreak("anything", []) == []


def test_chunks_without_a_section_path_are_handled():
    """Table chunks and windowed-fallback chunks can have no section at all."""
    results = apply_tiebreak(
        "waiting period", [chunk(0.50, ""), chunk(0.499, CLAUSE)]
    )
    assert results[0].section_path == CLAUSE


def test_match_counting_ignores_short_words():
    """"the", "for", "is" appear in half the section paths and carry no signal."""
    assert matches_section("what is the waiting period", chunk(0.5, CLAUSE)) >= 1
    assert matches_section("is the a of", chunk(0.5, CLAUSE)) == 0


def test_more_matching_terms_wins():
    broad = chunk(0.50, "4. Waiting Periods")
    specific = chunk(0.499, "4. Waiting Periods > 4.3 Pre-existing disease waiting period")
    results = apply_tiebreak("pre-existing disease waiting period", [broad, specific])
    assert results[0].chunk_id == specific.chunk_id
