from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Chunk, Document, DocumentStatus, Policy
from db.session import scoped, session_scope
from rag.config import get_settings
from rag.ingestion.pipeline import ingest_document
from services.api.deps import current_tenant_id, get_db
from services.api.schemas import DocumentOut, PolicyOut, UploadOut

router = APIRouter(prefix="/policies", tags=["policies"])

PDF_MAGIC = b"%PDF-"


def _validate_upload(data: bytes, filename: str) -> None:
    """Validate by content, not by extension -- the extension is attacker-controlled."""
    settings = get_settings()
    if len(data) == 0:
        raise HTTPException(400, "Empty file")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            413, f"File is {len(data)} bytes, above the {settings.max_upload_bytes} limit"
        )
    if not data.startswith(PDF_MAGIC):
        raise HTTPException(415, f"{filename} is not a PDF (missing %PDF- header)")


def _run_ingestion(document_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
    """Background entrypoint. Owns its own session: the request's is already closed."""
    with session_scope() as session:
        try:
            ingest_document(session, document_id, tenant_id)
        except Exception:
            # ingest_document has already recorded the failure on the document row.
            pass


@router.post("/upload", response_model=UploadOut)
async def upload_policy(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    product_name: str = Form(...),
    insurer: str = Form(""),
    policy_version: str = Form(""),
    session: Session = Depends(get_db),
) -> UploadOut:
    settings = get_settings()
    tenant_id = current_tenant_id(session)

    data = await file.read()
    _validate_upload(data, file.filename or "upload.pdf")

    checksum = hashlib.sha256(data).hexdigest()
    existing = session.scalar(
        scoped(select(Document).where(Document.checksum == checksum), Document, tenant_id)
    )
    if existing is not None:
        return UploadOut(
            policy_id=existing.policy_id,
            document_id=existing.id,
            status=existing.status,
            duplicate_of=existing.id,
        )

    policy = Policy(
        tenant_id=tenant_id,
        insurer=insurer,
        product_name=product_name,
        policy_version=policy_version,
    )
    session.add(policy)
    session.flush()

    storage_dir = Path(settings.storage_dir) / str(tenant_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = storage_dir / f"{checksum}.pdf"
    storage_path.write_bytes(data)

    document = Document(
        tenant_id=tenant_id,
        policy_id=policy.id,
        filename=file.filename or "upload.pdf",
        storage_path=str(storage_path),
        checksum=checksum,
        byte_size=len(data),
        status=DocumentStatus.PENDING,
    )
    session.add(document)
    session.commit()

    background.add_task(_run_ingestion, document.id, tenant_id)
    return UploadOut(policy_id=policy.id, document_id=document.id, status=document.status)


@router.post("/{policy_id}/process", response_model=list[DocumentOut])
def process_policy(
    policy_id: uuid.UUID,
    background: BackgroundTasks,
    force: bool = False,
    session: Session = Depends(get_db),
) -> list[DocumentOut]:
    """Trigger or inspect ingestion for a policy's documents."""
    tenant_id = current_tenant_id(session)
    documents = list(
        session.scalars(
            scoped(select(Document).where(Document.policy_id == policy_id), Document, tenant_id)
        )
    )
    if not documents:
        raise HTTPException(404, "No documents for this policy")

    out: list[DocumentOut] = []
    for document in documents:
        if force or document.status in (DocumentStatus.PENDING, DocumentStatus.FAILED):
            document.status = DocumentStatus.PENDING
            background.add_task(_run_ingestion, document.id, tenant_id)
        out.append(_document_out(session, document, tenant_id))
    session.commit()
    return out


def _document_out(session: Session, document: Document, tenant_id: uuid.UUID) -> DocumentOut:
    count = session.scalar(
        scoped(
            select(func.count(Chunk.id)).where(Chunk.document_id == document.id), Chunk, tenant_id
        )
    )
    return DocumentOut(
        id=document.id,
        filename=document.filename,
        status=document.status,
        page_count=document.page_count,
        chunk_count=int(count or 0),
        error=document.error,
    )


@router.get("", response_model=list[PolicyOut])
def list_policies(session: Session = Depends(get_db)) -> list[PolicyOut]:
    tenant_id = current_tenant_id(session)
    policies = list(session.scalars(scoped(select(Policy), Policy, tenant_id)))
    out: list[PolicyOut] = []
    for policy in policies:
        documents = list(
            session.scalars(
                scoped(
                    select(Document).where(Document.policy_id == policy.id), Document, tenant_id
                )
            )
        )
        out.append(
            PolicyOut(
                id=policy.id,
                insurer=policy.insurer,
                product_name=policy.product_name,
                policy_version=policy.policy_version,
                documents=[_document_out(session, d, tenant_id) for d in documents],
            )
        )
    return out
