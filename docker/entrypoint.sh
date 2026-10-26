#!/bin/sh
# Container start-up: schema first, demo policy second, server third.
#
# Migrations run here rather than in a separate release step because this image is the
# only thing that knows which revision the code expects. A server that boots against an
# older schema fails on the first query instead of at start-up, which is a worse place
# to find out.
set -e

echo "==> alembic upgrade head"
alembic upgrade head

echo "==> seeding the demo policy"
# Non-fatal on purpose. Seeding needs an OpenAI key for embeddings; if it is missing or
# the quota is gone, an API that starts with an empty library is far better than one that
# refuses to start at all.
python -m rag.cli seed-demo || \
  echo "WARNING: demo seeding failed; the library will start empty"

echo "==> uvicorn on :${PORT:-8000}"
exec uvicorn services.api.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers "${WEB_CONCURRENCY:-1}"
