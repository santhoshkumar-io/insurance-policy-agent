"""The evaluation set is a labelled dataset, so it gets validated like one."""

from __future__ import annotations

import pytest

from rag.evaluation.dataset import Answerability, load_eval_set

# Both suites are validated: the synthetic fixture and the headline benchmark.
SUITES = ["evals/questions.yaml", "evals/questions-lic.yaml"]


@pytest.mark.parametrize("suite", SUITES)
def test_eval_set_loads_and_validates(suite):
    eval_set = load_eval_set(suite)
    assert eval_set.questions
    assert eval_set.set_name


@pytest.mark.parametrize("suite", SUITES)
def test_every_answerable_question_has_a_retrieval_label(suite):
    """Unlabelled answerable questions silently depress Recall@k for no reason."""
    for question in load_eval_set(suite).answerable:
        assert question.has_retrieval_label, question.id


@pytest.mark.parametrize("suite", SUITES)
def test_unanswerable_questions_are_present_and_a_meaningful_share(suite):
    """Without them, abstention cannot be measured and the threshold cannot be calibrated."""
    eval_set = load_eval_set(suite)
    share = len(eval_set.unanswerable) / len(eval_set.questions)
    assert len(eval_set.unanswerable) >= 2
    assert 0.1 <= share <= 0.4, f"{suite}: unanswerable share is {share:.0%}"


@pytest.mark.parametrize("suite", SUITES)
def test_question_ids_are_unique(suite):
    questions = load_eval_set(suite).questions
    assert len({q.id for q in questions}) == len(questions)


@pytest.mark.parametrize("suite", SUITES)
def test_gold_answers_are_present_for_most_answerable_questions(suite):
    """A set where nothing is labelled cannot measure answer correctness later."""
    answerable = [q for q in load_eval_set(suite).answerable if not q.is_conditional]
    labelled = [q for q in answerable if q.has_answer_label]
    assert len(labelled) / len(answerable) > 0.8


def test_conditional_questions_label_several_pages():
    """The state exists because several competing conditions must be surfaced together."""
    for question in load_eval_set("evals/questions-lic.yaml").conditional:
        assert len(question.expected_pages) >= 2, question.id
        assert question.answerable is Answerability.CONDITIONAL
