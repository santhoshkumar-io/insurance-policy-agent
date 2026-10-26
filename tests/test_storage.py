"""Where a document's bytes live, and what happens when they do not.

These tests exist because of a deployment failure that no existing test could see: the
API wrote uploads to the local filesystem and recorded the path, which works on a
developer's machine and breaks on any host whose disk does not survive a restart. The
database keeps the row; the file goes away; `policy reindex` then fails on a document
that looks perfectly healthy in the library.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from db.models import Document, DocumentStatus, Policy, Tenant
from rag import storage
from rag.ingestion.pdf import extract_document
from rag.ingestion.pipeline import ingest_document

pytestmark = pytest.mark.db


def _document(session, tenant, policy, **kwargs) -> Document:
    document = Document(
        tenant_id=tenant.id,
        policy_id=policy.id,
        filename="policy.pdf",
        checksum=uuid.uuid4().hex * 2,
        status=DocumentStatus.PENDING,
        **kwargs,
    )
    session.add(document)
    session.flush()
    return document


@pytest.fixture
def tenant_and_policy(session):
    tenant = Tenant(slug=f"t-{uuid.uuid4().hex[:8]}", name="Storage Tenant")
    session.add(tenant)
    session.flush()
    policy = Policy(tenant_id=tenant.id, insurer="Acme", product_name="Health Guard")
    session.add(policy)
    session.flush()
    return tenant, policy


def test_a_document_ingests_from_bytes_with_no_file_on_disk(
    session, stub_openai, structured_policy_pdf, tenant_and_policy
):
    """The deployed path: nothing was ever written to a filesystem."""
    tenant, policy = tenant_and_policy
    document = _document(session, tenant, policy, storage_path=None)
    storage.store(document, structured_policy_pdf.read_bytes())
    session.flush()

    count = ingest_document(session, document.id, tenant.id)

    assert count > 0
    assert document.status == DocumentStatus.READY
    assert document.page_count == 4


def test_reindexing_survives_the_file_disappearing(
    session, stub_openai, structured_policy_pdf, tenant_and_policy, tmp_path
):
    """The redeploy case, reproduced.

    The document is ingested from a real file, the file is then deleted as a restart
    would delete it, and re-indexing must still work because the bytes are in the row.
    """
    tenant, policy = tenant_and_policy
    copy = tmp_path / "vanishing.pdf"
    copy.write_bytes(structured_policy_pdf.read_bytes())

    document = _document(session, tenant, policy, storage_path=str(copy))
    storage.store(document, copy.read_bytes())
    session.flush()
    first = ingest_document(session, document.id, tenant.id)

    copy.unlink()

    assert ingest_document(session, document.id, tenant.id) == first


def test_a_document_with_neither_bytes_nor_a_file_says_so(session, tenant_and_policy):
    """The pre-migration row whose file is gone: a clear message, not a FileNotFoundError."""
    tenant, policy = tenant_and_policy
    document = _document(session, tenant, policy, storage_path=None)

    with pytest.raises(storage.StorageError, match="re-uploaded"):
        storage.load(document)


def test_a_missing_file_names_the_deployment_cause(session, tenant_and_policy, tmp_path):
    tenant, policy = tenant_and_policy
    document = _document(session, tenant, policy, storage_path=str(tmp_path / "gone.pdf"))

    with pytest.raises(storage.StorageError, match="did not survive a restart"):
        storage.load(document)


def test_a_path_backed_document_still_loads(session, tenant_and_policy, structured_policy_pdf):
    """Rows written before the move keep working -- the migration carried no data step."""
    tenant, policy = tenant_and_policy
    document = _document(session, tenant, policy, storage_path=str(structured_policy_pdf))

    assert storage.load(document) == structured_policy_pdf.read_bytes()


@pytest.mark.parametrize("as_bytes", [False, True])
def test_extraction_agrees_whether_given_a_path_or_bytes(structured_policy_pdf, as_bytes):
    """The two entry points must not drift: same pages, same text, same checksum."""
    source = structured_policy_pdf.read_bytes() if as_bytes else structured_policy_pdf
    extracted = extract_document(source)
    reference = extract_document(structured_policy_pdf)

    assert extracted.page_count == reference.page_count
    assert extracted.checksum == reference.checksum
    assert [[s.text for s in p.spans] for p in extracted.pages] == [
        [s.text for s in p.spans] for p in reference.pages
    ]
    assert extracted.path == (None if as_bytes else Path(structured_policy_pdf))
