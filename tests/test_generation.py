"""Generation behaviour, driven by a stub model.

No network: these assert the control flow around the model -- abstention triggers,
citation rejection and the single bounded retry -- which is where the correctness
guarantees actually live.
"""

from __future__ import annotations

import pytest

from rag.config import get_settings
from rag.generation.answer import generate_answer
from tests.test_citation_validation import make_chunk


class StubModel:
    """Returns canned payloads in sequence and records what it was sent."""

    model = "stub-model"

    def __init__(self, payloads: list[dict]):
        self.payloads = list(payloads)
        self.calls: list[list[dict]] = []

    def complete(self, system, messages, schema):
        self.calls.append(messages)
        payload = self.payloads.pop(0) if self.payloads else {"answerable": False}
        return payload, {"prompt_tokens": 100, "completion_tokens": 20}


def answerable(chunk, text="Thirty six months.") -> dict:
    return {
        "answerable": True,
        "answer": text,
        "claims": [{"text": text, "citation_ids": [chunk.short_id]}],
        "insufficient_evidence_reason": None,
    }


def test_abstains_when_nothing_is_retrieved():
    answer = generate_answer("anything?", [], model=StubModel([]))
    assert not answer.answerable
    assert "No chunks retrieved" in answer.abstained_reason


def test_abstains_when_best_score_is_below_threshold():
    threshold = get_settings().abstain_score_threshold
    chunk = make_chunk(score=threshold - 0.05)
    answer = generate_answer("anything?", [chunk], model=StubModel([answerable(chunk)]))
    assert not answer.answerable
    assert "below the threshold" in answer.abstained_reason


def test_abstains_when_the_model_reports_insufficient_evidence():
    chunk = make_chunk()
    model = StubModel(
        [
            {
                "answerable": False,
                "answer": "",
                "claims": [],
                "insufficient_evidence_reason": "No clause about dental cover.",
            }
        ]
    )
    answer = generate_answer("is dental covered?", [chunk], model=model)
    assert not answer.answerable
    assert answer.abstained_reason == "No clause about dental cover."
    assert "can't answer that" in answer.text


def test_valid_answer_passes_through_with_citations():
    chunk = make_chunk()
    answer = generate_answer("waiting period?", [chunk], model=StubModel([answerable(chunk)]))
    assert answer.answerable
    assert answer.text == "Thirty six months."
    assert [c.chunk_id for c in answer.citations] == [chunk.chunk_id]
    assert answer.validation_retries == 0


def test_hallucinated_citation_triggers_one_retry_then_succeeds():
    chunk = make_chunk()
    bad = {
        "answerable": True,
        "answer": "Thirty six months.",
        "claims": [{"text": "Thirty six months.", "citation_ids": ["c_fabricated"]}],
        "insufficient_evidence_reason": None,
    }
    model = StubModel([bad, answerable(chunk)])
    answer = generate_answer("waiting period?", [chunk], model=model)

    assert answer.answerable
    assert answer.validation_retries == 1
    assert len(model.calls) == 2
    # The retry must name the violation, not just ask again.
    correction = model.calls[1][-1]["content"]
    assert "c_fabricated" in correction


def test_two_bad_attempts_abstain_rather_than_ship_an_unverifiable_citation():
    chunk = make_chunk()
    bad = {
        "answerable": True,
        "answer": "Covered in full.",
        "claims": [{"text": "Covered in full.", "citation_ids": ["c_nope"]}],
        "insufficient_evidence_reason": None,
    }
    answer = generate_answer("covered?", [chunk], model=StubModel([bad, bad]))
    assert not answer.answerable
    assert "not in the retrieved set" in answer.abstained_reason
    assert answer.citations == []


def test_uncited_claim_is_rejected():
    chunk = make_chunk()
    bare = {
        "answerable": True,
        "answer": "Probably covered.",
        "claims": [{"text": "Probably covered.", "citation_ids": []}],
        "insufficient_evidence_reason": None,
    }
    answer = generate_answer("covered?", [chunk], model=StubModel([bare, bare]))
    assert not answer.answerable


def test_usage_is_accumulated_across_the_retry():
    chunk = make_chunk()
    bad = {
        "answerable": True,
        "answer": "x",
        "claims": [{"text": "x", "citation_ids": ["c_bad"]}],
        "insufficient_evidence_reason": None,
    }
    answer = generate_answer("q?", [chunk], model=StubModel([bad, answerable(chunk)]))
    assert answer.usage["prompt_tokens"] == 200


@pytest.mark.parametrize("field", ["answerable", "answer", "claims"])
def test_schema_requires_the_fields_the_validator_depends_on(field):
    from rag.generation.prompt import ANSWER_SCHEMA

    assert field in ANSWER_SCHEMA["required"]
    assert ANSWER_SCHEMA["additionalProperties"] is False


def test_inline_citation_ids_are_stripped_from_the_answer_text():
    """Ids belong in the citation list; "(c_1a62c67c)" mid-sentence reads as a defect."""
    chunk = make_chunk()
    cid = chunk.short_id
    payload = {
        "answerable": True,
        "answer": f"The waiting period is thirty six months ({cid}).",
        "claims": [{"text": "Thirty six months.", "citation_ids": [cid]}],
        "insufficient_evidence_reason": None,
    }
    answer = generate_answer("waiting period?", [chunk], model=StubModel([payload]))

    assert cid not in answer.text
    assert answer.text == "The waiting period is thirty six months."
    # The citation itself survives -- only its appearance in the prose is removed.
    assert [c.chunk_id for c in answer.citations] == [chunk.chunk_id]


def test_inline_id_stripping_handles_the_shapes_models_actually_emit():
    from rag.generation.answer import strip_inline_ids

    assert strip_inline_ids("Covered (c_1a62c67c).") == "Covered."
    assert strip_inline_ids("Covered (source c_1a62c67c).") == "Covered."
    assert strip_inline_ids("Covered [c_1a62c67c, c_67c7a1b6].") == "Covered."
    assert strip_inline_ids("See c_1a62c67c for detail.") == "See for detail."
    # Text with no ids is left exactly as it was.
    assert strip_inline_ids("Thirty six months of cover.") == "Thirty six months of cover."
