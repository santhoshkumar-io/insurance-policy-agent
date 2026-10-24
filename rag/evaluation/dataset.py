"""Loading and validating the evaluation set.

The harness itself is week 3. This module exists now so the labels have a schema from
the start -- a question set that drifts in shape is a question set whose metrics cannot
be compared across weeks (plan.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PATH = Path("evals/questions.yaml")


@dataclass
class EvalQuestion:
    id: str
    question: str
    policy: str
    answerable: bool
    expected_pages: list[int] = field(default_factory=list)
    expected_section: str | None = None
    gold_answer: str | None = None
    notes: str | None = None

    @property
    def has_retrieval_label(self) -> bool:
        return bool(self.expected_pages) or self.expected_section is not None

    @property
    def has_answer_label(self) -> bool:
        return self.gold_answer is not None


@dataclass
class EvalSet:
    version: int
    set_name: str
    questions: list[EvalQuestion]

    @property
    def answerable(self) -> list[EvalQuestion]:
        return [q for q in self.questions if q.answerable]

    @property
    def unanswerable(self) -> list[EvalQuestion]:
        return [q for q in self.questions if not q.answerable]


def load_eval_set(path: Path | str = DEFAULT_PATH) -> EvalSet:
    data = yaml.safe_load(Path(path).read_text())
    questions = [EvalQuestion(**q) for q in data["questions"]]

    ids = [q.id for q in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate question ids in {path}: {sorted(duplicates)}")

    for question in questions:
        if question.answerable and not question.has_retrieval_label:
            raise ValueError(
                f"{question.id} is marked answerable but carries no expected evidence. "
                f"An answerable question without a retrieval label cannot be scored."
            )
        if not question.answerable and question.expected_pages:
            raise ValueError(
                f"{question.id} is marked unanswerable but lists expected pages."
            )

    return EvalSet(version=data["version"], set_name=data["set_name"], questions=questions)
