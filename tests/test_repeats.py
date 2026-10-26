"""Repeated generation passes, and the spread they report.

Finding 7 in docs/experiments.md measured one benchmark question answering 5 times and
abstaining 5 out of 10 -- at temperature 0, against identical evidence. A single
generation pass therefore cannot distinguish a real improvement from the coin landing
differently, which makes `--repeats` a prerequisite for the next phase rather than a
nicety. These tests pin the arithmetic so the published "± n" means what it says.
"""

from __future__ import annotations

from rag.evaluation.metrics import QuestionResult, aggregate


def _result(qid: str, answerable: bool, runs: list[bool], conditional: bool = False):
    return QuestionResult(
        question_id=qid,
        answerable=answerable,
        conditional=conditional,
        abstained=runs[0],
        abstained_runs=list(runs),
    )


def test_a_single_pass_reports_no_spread():
    """One observation says nothing about its own variance, so it claims nothing."""
    metrics = aggregate(
        "single",
        5,
        [
            _result("a1", True, [False]),
            _result("u1", False, [True]),
        ],
    )

    assert metrics.repeats == 1
    assert metrics.false_answer_spread is None
    assert metrics.false_abstention_spread is None
    assert metrics.as_row()["false answer"] == "0%"


def test_a_question_that_flips_shows_up_as_a_rate_with_a_band():
    """The L048 case: one unanswerable question, answered on half the passes."""
    metrics = aggregate(
        "flipper",
        5,
        [
            _result("a1", True, [False, False, False, False]),
            _result("u1", False, [True, False, True, False]),
        ],
    )

    assert metrics.repeats == 4
    # The rate is 0% on the passes it abstained and 100% on the passes it answered.
    assert metrics.false_answer_rate == 0.5
    assert metrics.false_answer_spread == 0.5
    assert metrics.as_row()["false answer"] == "50% ± 50"


def test_a_stable_configuration_reports_a_zero_band():
    """A band of zero is a claim worth making: it says the result is reproducible."""
    metrics = aggregate(
        "stable",
        5,
        [
            _result("a1", True, [False] * 5),
            _result("u1", False, [True] * 5),
        ],
    )

    assert metrics.false_answer_rate == 0.0
    assert metrics.false_answer_spread == 0.0
    assert metrics.as_row()["false abstain"] == "0% ± 0"


def test_conditional_questions_stay_out_of_the_repeated_rates():
    """They are excluded from the binary rates, and repeating them must not sneak them in."""
    metrics = aggregate(
        "conditional",
        5,
        [
            _result("a1", True, [False, False]),
            _result("c1", True, [True, True], conditional=True),
            _result("u1", False, [True, True]),
        ],
    )

    assert metrics.false_abstention_rate == 0.0


def test_the_rate_is_averaged_over_passes_not_over_questions():
    """Two questions, each failing on a different pass: the rate is 50% on both passes.

    Averaging per-question outcomes first would report the same 50% with no spread, and
    would hide a configuration that fails a *different* question every time.
    """
    metrics = aggregate(
        "alternating",
        5,
        [
            _result("u1", False, [False, True]),
            _result("u2", False, [True, False]),
        ],
    )

    assert metrics.false_answer_rate == 0.5
    assert metrics.false_answer_spread == 0.0
