"""Where a document's bytes live.

One module owns this decision, the way ``db.session.scoped`` owns tenant scoping, so the
day it changes there is one place to change it.

**The bytes go in the row.** Uploads used to be written to ``storage_dir`` and the path
recorded, which works on a developer's machine and fails on any host with an ephemeral
filesystem: the database survives a redeploy and the files do not, so ``policy reindex``
silently loses its source. Bytes in Postgres means one managed service instead of two, no
second set of credentials, backups that cover the documents as well as the rows, and a
file that commits in the same transaction as the row describing it.

**What that costs, so the trigger to change it is explicit.** A ``bytea`` column is stored
out of line by TOAST and compressed, so this is right for ~1MB wordings and a few hundred
documents. It is wrong in the tens of GB: the database is then paying for storage it is
bad at, backups get slow, and the answer is S3-compatible object storage behind these same
two functions. The other cost is quieter -- ``SELECT *`` on ``documents`` now drags every
PDF with it, so callers that do not need the file must name their columns.
"""

from __future__ import annotations

from pathlib import Path

from db.models import Document


class StorageError(RuntimeError):
    pass


def store(document: Document, data: bytes) -> None:
    """Attach the file to the document row."""
    document.content = data
    document.byte_size = len(data)


def load(document: Document) -> bytes:
    """Return the document's bytes.

    Falls back to ``storage_path`` for rows written before the move, and for documents
    the CLI ingested from a file it had no reason to copy. A missing file is reported as
    a storage failure rather than a ``FileNotFoundError`` from three frames deeper.
    """
    if document.content is not None:
        return document.content

    if not document.storage_path:
        raise StorageError(
            f"Document {document.id} has neither stored bytes nor a storage path. "
            f"It was probably ingested before the file was kept, and must be re-uploaded."
        )

    path = Path(document.storage_path)
    if not path.exists():
        raise StorageError(
            f"Document {document.id} points at {path}, which does not exist. "
            f"If this is a deployed instance, the filesystem did not survive a restart "
            f"-- re-upload the document so its bytes are stored in the database."
        )
    return path.read_bytes()
