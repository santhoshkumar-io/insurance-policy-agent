from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class PolicyOut(BaseModel):
    id: uuid.UUID
    insurer: str
    product_name: str
    policy_version: str
    documents: list[DocumentOut] = Field(default_factory=list)


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    status: str
    page_count: int
    chunk_count: int = 0
    error: str | None = None


class UploadOut(BaseModel):
    policy_id: uuid.UUID
    document_id: uuid.UUID
    status: str
    duplicate_of: uuid.UUID | None = None


class CitationOut(BaseModel):
    chunk_id: uuid.UUID
    ref: str
    document: str
    policy: str
    page_from: int
    page_to: int
    section_path: str
    chunk_type: str
    score: float
    text: str


class ClaimOut(BaseModel):
    text: str
    citation_refs: list[str]


class ChatIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    policy_ids: list[uuid.UUID] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class ChatOut(BaseModel):
    answerable: bool
    answer: str
    abstained_reason: str | None = None
    claims: list[ClaimOut] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    retrieved: list[CitationOut] = Field(default_factory=list)
    model: str
    prompt_version: str
    latency_ms: int
    message_id: uuid.UUID | None = None


PolicyOut.model_rebuild()
