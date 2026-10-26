# The API image. One process, no filesystem state: uploaded PDFs live in the database
# (rag/storage.py), which is what lets this run on a host whose disk is ephemeral.
FROM python:3.12-slim

# Pinned rather than :latest -- a build that silently changes its installer is a build
# that cannot be reproduced.
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # seed-demo imports scripts/sample_policy.py, which is a script rather than part of
    # the installed package. Without this the console entry point cannot find it.
    PYTHONPATH=/app

WORKDIR /app

# Source before install: hatchling builds the wheel from these directories, so they have
# to be present. The [rerank] extra is deliberately not installed -- it pulls in torch,
# for a cross-encoder the benchmark measured and rejected as a default.
COPY pyproject.toml README.md alembic.ini ./
COPY rag ./rag
COPY services ./services
COPY db ./db
COPY scripts ./scripts
RUN uv pip install --system --no-cache .

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Run as nobody in particular. The container writes nothing to disk; if it starts trying
# to, that is a bug worth failing on rather than a permission to grant.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["/usr/local/bin/entrypoint.sh"]
