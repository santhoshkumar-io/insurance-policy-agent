"""The evaluation set is a labelled dataset, so it gets validated like one."""

from __future__ import annotations

from rag.evaluation.dataset import load_eval_set


def test_eval_set_loads_and_validates():
    eval_set = load_eval_set()
    assert eval_set.questions
    assert eval_set.set_name


def test_every_answerable_question_has_a_retrieval_label():
    """Unlabelled answerable questions silently depress Recall@k for no reason."""
    for question in load_eval_set().answerable:
        assert question.has_retrieval_label, question.id


def test_unanswerable_questions_are_present_and_a_meaningful_share():
    """Without them, abstention cannot be measured and the threshold cannot be calibrated."""
    eval_set = load_eval_set()
    share = len(eval_set.unanswerable) / len(eval_set.questions)
    assert len(eval_set.unanswerable) >= 2
    assert 0.1 <= share <= 0.4, f"unanswerable share is {share:.0%}"


def test_question_ids_are_unique():
    questions = load_eval_set().questions
    assert len({q.id for q in questions}) == len(questions)
