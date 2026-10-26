"""Add a generated tsvector over section_path, for the section-aware retrieval arm.

Mirrors 0002: generated rather than trigger-maintained, so it cannot drift from the
column it indexes and existing rows are populated by the ALTER itself.

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN section_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', section_path)) STORED
        """
    )
    op.execute("CREATE INDEX ix_chunks_section_tsv ON chunks USING gin (section_tsv)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_section_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS section_tsv")
