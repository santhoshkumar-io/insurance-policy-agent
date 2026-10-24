"""Grounded answer generation with enforced citations and an explicit abstention path."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from rag.config import get_settings
from rag.generation.prompt import (
    ANSWER_SCHEMA,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_message,
)
from rag.retrieval.types import RetrievedChunk
from rag.validation.citations import Claim, validate_claims

ABSTAIN_TEXT = (
    "I can't answer that from the selected document(s). The retrieved text does not "
    "contain the relevant wording."
)

# The prompt forbids inline ids, but a model will occasionally write one anyway, and
# "(c_1a62c67c)" in the middle of a sentence reads as a defect to the agent. The ids
# are already rendered properly from the citation list, so strip any that leak through.
_INLINE_ID_GROUP = re.compile(
    r"\s*[(\[]\s*(?:sources?:?\s*)?c_[0-9a-f]{8}(?:\s*,\s*c_[0-9a-f]{8})*\s*[)\]]"
)
_INLINE_ID_BARE = re.compile(r"\s*\b(?:sources?:?\s*)?c_[0-9a-f]{8}\b")


def strip_inline_ids(text: str) -> str:
    text = _INLINE_ID_GROUP.sub("", text)
    text = _INLINE_ID_BARE.sub("", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


class ChatModel(Protocol):
    """Minimal surface the generator needs, so tests can supply a stub."""

    def complete(self, system: str, messages: list[dict], schema: dict) -> tuple[dict, dict]: ...


@dataclass
class CitedClaim:
    text: str
    chunk_ids: list[str]


@dataclass
class Answer:
    answerable: bool
    text: str
    claims: list[CitedClaim] = field(default_factory=list)
    citations: list[RetrievedChunk] = field(default_factory=list)
    abstained_reason: str | None = None
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    latency_ms: int = 0
    usage: dict = field(default_factory=dict)
    validation_retries: int = 0


class OpenAIChatModel:
    def __init__(self, model: str | None = None):
        self.model = model or get_settings().openai_chat_model

    def complete(self, system: str, messages: list[dict], schema: dict) -> tuple[dict, dict]:
        from rag.embeddings.openai_embeddings import get_client

        response = get_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "policy_answer", "strict": True, "schema": schema},
            },
            temperature=0,
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        usage = response.usage.model_dump() if response.usage else {}
        return payload, usage


def _abstain(reason: str, model: str, started: float) -> Answer:
    return Answer(
        answerable=False,
        text=ABSTAIN_TEXT,
        abstained_reason=strip_inline_ids(reason),
        model=model,
        latency_ms=int((time.time() - started) * 1000),
    )


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    model: ChatModel | None = None,
) -> Answer:
    """Answer the question from the retrieved chunks, or abstain.

    Abstention has three triggers, all explicit (plan.md §2.7):
      - nothing was retrieved;
      - the best retrieval score falls below the calibrated threshold;
      - the model itself reports the evidence is insufficient.

    A fourth path rejects the answer: if the model cites an id it was never given, it is
    told so and asked once more. A second failure abstains rather than shipping an
    unverifiable citation.
    """
    settings = get_settings()
    model = model or OpenAIChatModel()
    model_name = getattr(model, "model", settings.openai_chat_model)
    started = time.time()

    if not chunks:
        return _abstain("No chunks retrieved for this query and policy scope.", model_name, started)

    # RetrievedChunk.confidence is a cosine similarity whatever ranked the list, so this
    # threshold means the same thing under vector, hybrid and reranked retrieval. Reading
    # `score` here instead would compare 0.25 against an RRF sum of ~0.016 and abstain on
    # every question (docs/decisions.md, the score-scale trap).
    best = max(c.confidence for c in chunks)
    if best < settings.abstain_semantic_threshold:
        return _abstain(
            f"Best evidence similarity {best:.3f} is below the threshold "
            f"{settings.abstain_semantic_threshold:.3f}.",
            model_name,
            started,
        )

    messages = [{"role": "user", "content": build_user_message(question, chunks)}]
    usage_total: dict = {}
    retries = 0

    for attempt in range(2):
        payload, usage = model.complete(SYSTEM_PROMPT, messages, ANSWER_SCHEMA)
        for key, value in usage.items():
            if isinstance(value, int):
                usage_total[key] = usage_total.get(key, 0) + value

        if not payload.get("answerable", False):
            answer = _abstain(
                payload.get("insufficient_evidence_reason")
                or "Model reported insufficient evidence.",
                model_name,
                started,
            )
            answer.usage = usage_total
            answer.validation_retries = retries
            return answer

        claims = [
            Claim(text=c.get("text", ""), citation_ids=list(c.get("citation_ids", [])))
            for c in payload.get("claims", [])
        ]
        result = validate_claims(claims, chunks)

        if result.ok:
            by_id = {c.short_id: c for c in chunks}
            cited = [by_id[cid] for cid in result.used_ids if cid in by_id]
            return Answer(
                answerable=True,
                text=strip_inline_ids(payload.get("answer", "")),
                claims=[CitedClaim(text=c.text, chunk_ids=c.citation_ids) for c in claims],
                citations=cited,
                model=model_name,
                latency_ms=int((time.time() - started) * 1000),
                usage=usage_total,
                validation_retries=retries,
            )

        if attempt == 0:
            retries = 1
            messages += [
                {"role": "assistant", "content": json.dumps(payload)},
                {
                    "role": "user",
                    "content": (
                        f"That response failed citation validation. {result.failure_message()} "
                        "Answer again using only the ids listed earlier, and give every claim "
                        "at least one valid citation. If you cannot support a claim with the "
                        "sources provided, set answerable to false."
                    ),
                },
            ]

    answer = _abstain(
        "Model produced citations that were not in the retrieved set, twice.",
        model_name,
        started,
    )
    answer.usage = usage_total
    answer.validation_retries = retries
    return answer
