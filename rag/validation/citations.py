"""Citation validation.

The brief requires that citations point to retrieved chunks rather than being generated
as free-form page guesses. That is an outcome; this module is the mechanism (plan.md §2.6).

What is guaranteed here is *existence*: every id the model returned was actually served
to it, and page numbers come from the database record rather than from model output.
Whether the model cited the *right* real clause is a question for evaluation -- no
validator can answer it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.retrieval.vector import RetrievedChunk


@dataclass
class Claim:
    text: str
    citation_ids: list[str]


@dataclass
class ValidationResult:
    ok: bool
    invalid_ids: list[str] = field(default_factory=list)
    uncited_claims: list[str] = field(default_factory=list)
    used_ids: list[str] = field(default_factory=list)

    def failure_message(self) -> str:
        parts = []
        if self.invalid_ids:
            parts.append(
                "These citation ids were not among the sources provided: "
                + ", ".join(sorted(set(self.invalid_ids)))
            )
        if self.uncited_claims:
            parts.append(
                f"{len(self.uncited_claims)} claim(s) carried no citation at all."
            )
        return " ".join(parts)


def validate_claims(claims: list[Claim], served: list[RetrievedChunk]) -> ValidationResult:
    served_ids = {c.short_id for c in served}
    invalid: list[str] = []
    uncited: list[str] = []
    used: list[str] = []

    for claim in claims:
        if not claim.citation_ids:
            uncited.append(claim.text)
            continue
        for cid in claim.citation_ids:
            if cid in served_ids:
                used.append(cid)
            else:
                invalid.append(cid)

    return ValidationResult(
        ok=not invalid and not uncited,
        invalid_ids=invalid,
        uncited_claims=uncited,
        used_ids=list(dict.fromkeys(used)),
    )
