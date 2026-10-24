"""Add a generated tsvector column and its GIN index for the keyword arm.

Generated rather than trigger-maintained: PostgreSQL keeps it in step with `text`
automatically, so it cannot drift, and existing rows are populated by the ALTER itself
with no backfill script to forget to run.

This is PostgreSQL full-text search, not BM25 -- see rag/retrieval/keyword.py.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN text_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
        """
    )
    op.execute("CREATE INDEX ix_chunks_text_tsv ON chunks USING gin (text_tsv)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_text_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS text_tsv")
