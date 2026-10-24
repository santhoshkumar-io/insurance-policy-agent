"""Draft candidate benchmark questions from an ingested policy.

Generation is cheap; verification is the real cost and cannot be delegated. So this
produces *candidates with their source page attached* and nothing more — a human reads
each one against the document and either writes the label or drops the question.

What it deliberately does not do is write `evals/questions.yaml`. A benchmark that
labelled itself would measure the model's agreement with itself, and every number built
on it would be worthless.
"""

from __future__ import annotations

import json
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Chunk
from db.session import scoped
from rag.config import get_settings

DRAFT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["questions"],
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["question", "answer_in_source", "why_hard"],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What an insurance agent would actually ask.",
                    },
                    "answer_in_source": {
                        "type": "string",
                        "description": "The exact wording from the source that answers it.",
                    },
                    "why_hard": {
                        "type": "string",
                        "description": "What makes this a useful test case.",
                    },
                },
            },
        }
    },
}

SYSTEM = """\
You draft candidate questions for a retrieval benchmark over insurance policy documents.

You are given one extract from a policy. Propose questions an agent would genuinely ask \
whose answer is contained in that extract.

Make them hard in ways that matter:
- paraphrase rather than reusing the document's own words, so the test is not a string match;
- target a specific figure, period or condition rather than a whole section;
- where the extract is a table, ask for one cell's value together with its row label, \
since the row-to-column relationship is what naive extraction destroys.

Do not invent facts. Every question must be answerable from this extract alone, and \
"answer_in_source" must quote the extract. Propose at most three questions, fewer if the \
extract does not support three good ones."""


def draft_for_policy(
    session: Session,
    tenant_id: uuid.UUID,
    policy_id: uuid.UUID,
    per_chunk: int = 3,
    limit: int | None = None,
) -> list[dict]:
    """Return candidate questions, each tagged with the page it came from."""
    from rag.embeddings.openai_embeddings import get_client

    settings = get_settings()
    client = get_client()

    stmt = scoped(
        select(Chunk).where(Chunk.policy_id == policy_id).order_by(Chunk.ordinal),
        Chunk,
        tenant_id,
    )
    if limit:
        stmt = stmt.limit(limit)
    chunks = list(session.scalars(stmt))

    candidates: list[dict] = []
    for chunk in chunks:
        extract = (
            f"[page {chunk.page_from}"
            + (f"-{chunk.page_to}" if chunk.page_to != chunk.page_from else "")
            + f", section: {chunk.section_path or 'none'}, type: {chunk.chunk_type}]\n\n"
            + chunk.text
        )
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": extract},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "question_candidates",
                    "strict": True,
                    "schema": DRAFT_SCHEMA,
                },
            },
            temperature=0.3,
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        for item in payload.get("questions", [])[:per_chunk]:
            candidates.append(
                {
                    "question": item.get("question", ""),
                    "answer_in_source": item.get("answer_in_source", ""),
                    "why_hard": item.get("why_hard", ""),
                    "expected_pages": sorted({chunk.page_from, chunk.page_to}),
                    "expected_section": chunk.section_path or None,
                    "chunk_type": chunk.chunk_type,
                    "source_chunk": str(chunk.id),
                }
            )
    return candidates


def to_yaml_stub(candidates: list[dict], policy_label: str, start_index: int = 1) -> str:
    """Render candidates as YAML entries, each marked UNVERIFIED.

    The marker is the point. Anything still carrying it has not been checked against the
    document and must not be counted in a published metric.
    """
    lines = [
        "# CANDIDATES -- NOT YET A BENCHMARK.",
        "# Every entry below was proposed by a model and verified by nobody. Read each one",
        "# against the source document, correct or delete it, then remove the UNVERIFIED",
        "# marker. Entries still marked UNVERIFIED must not appear in a published metric.",
        "",
    ]
    for offset, candidate in enumerate(candidates):
        number = start_index + offset
        pages = ", ".join(str(p) for p in candidate["expected_pages"])
        section = candidate["expected_section"]
        lines += [
            f"  - id: c{number:03d}",
            f"    question: {json.dumps(candidate['question'])}",
            f"    policy: {policy_label}",
            "    answerable: true",
            f"    expected_pages: [{pages}]",
            f"    expected_section: {json.dumps(section) if section else 'null'}",
            "    gold_answer: null",
            f"    notes: \"UNVERIFIED -- {candidate['why_hard']}\"",
            f"    # source says: {candidate['answer_in_source'][:160]}",
            "",
        ]
    return "\n".join(lines)
