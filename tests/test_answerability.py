"""Tri-state answerability.

A boolean could not express the honest answer to the brief's own example question --
"Does this policy cover a surgery after six months?" -- whose answer depends on facts the
question does not supply. Scored as a binary, the system's defensible abstention counted
as an error.
"""

from __future__ import annotations

import pytest

from rag.evaluation.dataset import Answerability
from rag.evaluation.metrics import QuestionResult, aggregate


def test_legacy_booleans_still_parse():
    """The synthetic suite is written with true/false and must keep working.

    Rewriting a working regression fixture to prove a point would be a poor trade.
    """
    assert Answerability.parse(True) is Answerability.YES
    assert Answerability.parse(False) is Answerability.NO


def test_tri_state_strings_parse():
    assert Answerability.parse("yes") is Answerability.YES
    assert Answerability.parse("no") is Answerability.NO
    assert Answerability.parse("conditional") is Answerability.CONDITIONAL
    assert Answerability.parse("CONDITIONAL") is Answerability.CONDITIONAL


def test_an_unknown_state_is_rejected_rather_than_coerced():
    with pytest.raises(ValueError):
        Answerability.parse("maybe")
    with pytest.raises(ValueError):
        Answerability.parse(None)


def test_conditional_questions_are_excluded_from_both_binary_rates():
    """Neither answering nor abstaining is wrong when the answer is "it depends"."""
    results = [
        # A conditional question that abstained: must NOT count as a false abstention.
        QuestionResult(
            "c1", answerable=True, conditional=True, abstained=True, conditions_covered=True
        ),
        QuestionResult("q1", answerable=True, abstained=False),
        QuestionResult("q2", answerable=False, abstained=True),
    ]
    metrics = aggregate("test", k=5, results=results)
    assert metrics.false_abstention_rate == 0.0
    assert metrics.false_answer_rate == 0.0


def test_a_conditional_abstention_would_have_been_an_error_under_the_old_scheme():
    """Demonstrates what the state fixes: the same data, counted as a binary."""
    results = [
        QuestionResult("c1", answerable=True, conditional=False, abstained=True),
        QuestionResult("q1", answerable=True, abstained=False),
    ]
    binary = aggregate("test", k=5, results=results)
    assert binary.false_abstention_rate == 0.5  # the old, misleading number

    results[0].conditional = True
    tri = aggregate("test", k=5, results=results)
    assert tri.false_abstention_rate == 0.0


def test_conditional_coverage_requires_every_condition():
    """Surfacing one of three competing conditions is the failure this catches."""
    results = [
        QuestionResult("c1", answerable=True, conditional=True, conditions_covered=True),
        QuestionResult("c2", answerable=True, conditional=True, conditions_covered=False),
    ]
    metrics = aggregate("test", k=5, results=results)
    assert metrics.conditional_coverage == 0.5


def test_conditional_coverage_is_none_when_the_set_has_no_conditional_questions():
    """An em dash in the table is honest; 0% would read as a measured failure."""
    results = [QuestionResult("q1", answerable=True)]
    metrics = aggregate("test", k=5, results=results)
    assert metrics.conditional_coverage is None
    assert metrics.as_row()["cond. coverage"] == "—"


def test_conditional_questions_still_count_toward_retrieval_quality():
    """There is specific evidence to find, so recall and MRR remain meaningful."""
    results = [
        QuestionResult("c1", answerable=True, conditional=True, recall=1.0, rr=1.0),
        QuestionResult("q1", answerable=True, recall=0.0, rr=0.0),
    ]
    metrics = aggregate("test", k=5, results=results)
    assert metrics.recall_at_k == 0.5
    assert metrics.answerable == 2
