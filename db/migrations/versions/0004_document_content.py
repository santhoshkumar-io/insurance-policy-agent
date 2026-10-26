"""Store the PDF bytes alongside the document row.

Uploaded files were written to the local filesystem and the path recorded. That works on
a developer's machine and fails on any host with an ephemeral disk: the database survives
a redeploy, the files do not, and ``policy reindex`` -- which re-reads the original PDF --
stops working with no visible error until someone re-indexes.

``storage_path`` becomes nullable rather than being dropped, and no data is migrated.
Existing rows keep their path and keep working, which is what makes this safe to run
against a database that already has documents in it.

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("content", sa.LargeBinary(), nullable=True))
    op.alter_column("documents", "storage_path", existing_type=sa.String(1024), nullable=True)


def downgrade() -> None:
    # Rows stored only as bytes have no path to fall back to, so they would violate the
    # restored NOT NULL. Give them a marker rather than failing the downgrade halfway.
    op.execute("UPDATE documents SET storage_path = '' WHERE storage_path IS NULL")
    op.alter_column("documents", "storage_path", existing_type=sa.String(1024), nullable=False)
    op.drop_column("documents", "content")
