from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from db.models import Citation, Conversation, Message
from rag.config import get_settings
from rag.generation.answer import Answer, generate_answer
from rag.retrieval.search import search
from rag.retrieval.types import RetrievedChunk
from services.api.deps import current_tenant_id, get_db
from services.api.limits import enforce_daily_cap, enforce_ip_rate_limit
from services.api.schemas import ChatIn, ChatOut, CitationOut, ClaimOut

router = APIRouter(tags=["chat"])


def citation_out(chunk: RetrievedChunk) -> CitationOut:
    """Render a citation from the database record.

    Page and section come from the chunk row, never from model output -- that is what
    makes the citation verifiable rather than asserted.
    """
    return CitationOut(
        chunk_id=chunk.chunk_id,
        ref=chunk.short_id,
        document=chunk.document_name,
        policy=chunk.policy_name,
        page_from=chunk.page_from,
        page_to=chunk.page_to,
        section_path=chunk.section_path,
        chunk_type=chunk.chunk_type,
        score=round(chunk.score, 4),
        semantic_score=(
            round(chunk.semantic_score, 4) if chunk.semantic_score is not None else None
        ),
        retrieved_by=str(chunk.retrieved_by),
        text=chunk.text,
    )


def _persist(
    session: Session, tenant_id, question: str, answer: Answer, retrieved: list[RetrievedChunk]
) -> Message:
    conversation = Conversation(tenant_id=tenant_id, title=question[:200])
    session.add(conversation)
    session.flush()

    message = Message(
        tenant_id=tenant_id,
        conversation_id=conversation.id,
        question=question,
        answer=answer.text,
        answerable=answer.answerable,
        abstained_reason=answer.abstained_reason,
        model=answer.model,
        prompt_version=answer.prompt_version,
        latency_ms=answer.latency_ms,
        usage=answer.usage or None,
        # The audit trail the brief asks for: chunk ids and model/prompt version,
        # not the document text itself.
        retrieved_chunk_ids=[str(c.chunk_id) for c in retrieved],
    )
    session.add(message)
    session.flush()

    for chunk in answer.citations:
        session.add(
            Citation(
                tenant_id=tenant_id,
                message_id=message.id,
                chunk_id=chunk.chunk_id,
                score=chunk.score,
            )
        )
    session.flush()
    return message


@router.post("/chat", response_model=ChatOut)
def chat(
    payload: ChatIn, request: Request, session: Session = Depends(get_db)
) -> ChatOut:
    # The demo is open, so this endpoint is an unmetered path to a paid key unless it is
    # capped. Both checks raise 429 with a message the dashboard shows the reader.
    enforce_ip_rate_limit(request, get_settings().rate_limit_questions_per_minute)
    enforce_daily_cap(session)

    tenant_id = current_tenant_id(session)

    retrieved = search(
        session,
        tenant_id,
        payload.question,
        policy_ids=payload.policy_ids,
        top_k=payload.top_k,
        mode=payload.mode,
    )
    answer = generate_answer(payload.question, retrieved)
    message = _persist(session, tenant_id, payload.question, answer, retrieved)
    session.commit()

    return ChatOut(
        answerable=answer.answerable,
        answer=answer.text,
        abstained_reason=answer.abstained_reason,
        claims=[ClaimOut(text=c.text, citation_refs=c.chunk_ids) for c in answer.claims],
        citations=[citation_out(c) for c in answer.citations],
        retrieved=[citation_out(c) for c in retrieved],
        model=answer.model,
        prompt_version=answer.prompt_version,
        latency_ms=answer.latency_ms,
        message_id=message.id,
    )
