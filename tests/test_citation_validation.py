"""The citation guarantee, tested directly.

A model that invents 'c_deadbeef' must never reach the user with it (plan.md §2.6).
"""

from __future__ import annotations

import uuid

from rag.retrieval.vector import RetrievedChunk
from rag.validation.citations import Claim, validate_claims


def make_chunk(text: str = "clause text", score: float = 0.8) -> RetrievedChunk:
    cid = uuid.uuid4()
    return RetrievedChunk(
        chunk_id=cid,
        text=text,
        page_from=2,
        page_to=2,
        section_path="4. Waiting Periods",
        chunk_type="prose",
        policy_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_name="policy.pdf",
        policy_name="Health Guard",
        score=score,
    )


def test_accepts_claims_citing_served_chunks():
    chunk = make_chunk()
    result = validate_claims([Claim("36 months", [chunk.short_id])], [chunk])
    assert result.ok
    assert result.used_ids == [chunk.short_id]


def test_rejects_an_invented_citation_id():
    chunk = make_chunk()
    result = validate_claims([Claim("36 months", ["c_deadbeef"])], [chunk])
    assert not result.ok
    assert result.invalid_ids == ["c_deadbeef"]
    assert "c_deadbeef" in result.failure_message()


def test_rejects_a_claim_with_no_citation_at_all():
    chunk = make_chunk()
    result = validate_claims([Claim("the policy is generous", [])], [chunk])
    assert not result.ok
    assert result.uncited_claims == ["the policy is generous"]


def test_partial_validity_still_fails():
    """One good citation does not launder a bad one in the same answer."""
    chunk = make_chunk()
    result = validate_claims(
        [Claim("a", [chunk.short_id]), Claim("b", ["c_notreal"])], [chunk]
    )
    assert not result.ok
    assert result.invalid_ids == ["c_notreal"]


def test_short_ids_are_stable_and_distinct():
    chunks = [make_chunk() for _ in range(20)]
    refs = [c.short_id for c in chunks]
    assert len(set(refs)) == len(refs)
    assert all(r.startswith("c_") for r in refs)
