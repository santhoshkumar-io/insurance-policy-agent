"""Loading and validating the evaluation set.

The harness itself is week 3. This module exists now so the labels have a schema from
the start -- a question set that drifts in shape is a question set whose metrics cannot
be compared across weeks (plan.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

DEFAULT_PATH = Path("evals/questions.yaml")


class Answerability(StrEnum):
    """Whether the document can answer the question at all.

    ``CONDITIONAL`` exists because a boolean could not express the honest answer to the
    brief's own example question -- "Does this policy cover a surgery after six months?"
    The document holds three different waiting periods and cannot say which applies
    without facts the question does not supply, so the right response is "it depends, and
    here they are". Scored as a binary, the system's entirely defensible abstention was
    counted as an error and showed up as a 6% false-abstention rate.

    Conditional questions are excluded from the false-answer and false-abstention rates
    and scored instead on whether retrieval surfaced *every* condition.
    """

    YES = "yes"
    NO = "no"
    CONDITIONAL = "conditional"

    @classmethod
    def parse(cls, value) -> Answerability:
        """Accept the tri-state strings and the legacy booleans.

        The synthetic suite was written with `answerable: true/false`, and rewriting it
        to prove a point would discard a working regression fixture.
        """
        if isinstance(value, bool):
            return cls.YES if value else cls.NO
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        if text in {"true", "yes"}:
            return cls.YES
        if text in {"false", "no"}:
            return cls.NO
        if text == "conditional":
            return cls.CONDITIONAL
        raise ValueError(
            f"answerable must be true, false or 'conditional', not {value!r}"
        )


@dataclass
class EvalQuestion:
    id: str
    question: str
    policy: str
    answerable: Answerability
    expected_pages: list[int] = field(default_factory=list)
    expected_section: str | None = None
    gold_answer: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        self.answerable = Answerability.parse(self.answerable)

    @property
    def has_retrieval_label(self) -> bool:
        return bool(self.expected_pages) or self.expected_section is not None

    @property
    def has_answer_label(self) -> bool:
        return self.gold_answer is not None

    @property
    def is_answerable(self) -> bool:
        """Scored on retrieval quality: there is specific evidence to find."""
        return self.answerable in {Answerability.YES, Answerability.CONDITIONAL}

    @property
    def is_conditional(self) -> bool:
        return self.answerable is Answerability.CONDITIONAL

    @property
    def is_unanswerable(self) -> bool:
        return self.answerable is Answerability.NO


@dataclass
class EvalSet:
    version: int
    set_name: str
    questions: list[EvalQuestion]

    @property
    def answerable(self) -> list[EvalQuestion]:
        """Questions with evidence to find, conditional ones included."""
        return [q for q in self.questions if q.is_answerable]

    @property
    def unanswerable(self) -> list[EvalQuestion]:
        return [q for q in self.questions if q.is_unanswerable]

    @property
    def conditional(self) -> list[EvalQuestion]:
        return [q for q in self.questions if q.is_conditional]


def load_eval_set(path: Path | str = DEFAULT_PATH) -> EvalSet:
    data = yaml.safe_load(Path(path).read_text())
    questions = [EvalQuestion(**q) for q in data["questions"]]

    ids = [q.id for q in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate question ids in {path}: {sorted(duplicates)}")

    for question in questions:
        if question.is_answerable and not question.has_retrieval_label:
            raise ValueError(
                f"{question.id} is marked answerable but carries no expected evidence. "
                f"An answerable question without a retrieval label cannot be scored."
            )
        if question.is_unanswerable and question.expected_pages:
            raise ValueError(
                f"{question.id} is marked unanswerable but lists expected pages."
            )
        if question.is_conditional and len(question.expected_pages) < 2:
            raise ValueError(
                f"{question.id} is conditional but labels fewer than two pages. The "
                f"point of the state is that several passages must be surfaced together; "
                f"with one page it is an ordinary answerable question."
            )

    return EvalSet(version=data["version"], set_name=data["set_name"], questions=questions)
