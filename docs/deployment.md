# Deployment

Three services, each on a free tier, none of them holding state the others need:

| Piece | Where | Why there |
|---|---|---|
| Postgres + pgvector | **Neon** | pgvector is available; the storage decision means it holds the PDFs too |
| API (Docker) | **Render** | runs the repository's `Dockerfile` unmodified |
| Dashboard | **Vercel** | where Next.js belongs |

Nothing here is load-bearing. The API is a plain container with one environment
contract, so moving it to Fly, Railway or a VM is a redeploy, not a rewrite.

## 1. Database

Create a Neon project and enable the extension once:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Take the **pooled** connection string — the host containing `-pooler` — and use it as
`DATABASE_URL` exactly as Neon gives it to you:

```
postgresql://user:password@ep-xxx-pooler.region.aws.neon.tech/neondb?sslmode=require
```

No hand-editing. `rag/config.py` pins the driver to psycopg 3 itself, because which
driver a bare `postgresql://` selects is a SQLAlchemy default that changed between 2.0
and 2.1 — and a secret that has to be edited by hand before it works is its own small
source of outages.

Pooled rather than direct because the API holds a connection pool per worker and Neon's
direct endpoint has a much lower connection ceiling. Migrations run over the same URL at
container start.

Migrations are **not** run by hand. The container runs `alembic upgrade head` on every
boot, because the image is the only thing that knows which revision its code expects.

## 2. API

A Render Web Service from this repository, runtime **Docker**, health check path
**`/ready`** — not `/health`. `/health` answers while the database is unreachable, which
is the point of having both (see `docs/decisions.md`).

Environment:

```
OPENAI_API_KEY=sk-...
DATABASE_URL=<the pooled URI from step 1>
CORS_ORIGINS=["https://<your-app>.vercel.app"]
RATE_LIMIT_QUESTIONS_PER_MINUTE=10
RATE_LIMIT_UPLOADS_PER_MINUTE=3
DAILY_QUESTION_CAP=300
```

`CORS_ORIGINS` must contain the dashboard's real origin — a JSON array is the documented
form, but a bare URL or a comma-separated list is accepted too, and a trailing slash is
stripped. That tolerance is not politeness: settings are read at import time, so a value
the parser rejects does not misconfigure CORS, it stops the container booting.

It is never `*`, and `tests/test_cors.py` fails if it becomes one.

**Vercel preview deployments get their own URLs** (`...-git-branch-....vercel.app`) and
will not be in this list. Only the production domain works unless you add them.

On first boot the container migrates, seeds the synthetic demo policy, and starts
uvicorn. Seeding is non-fatal: without a working key the API still starts, with an empty
library.

**A free instance sleeps after inactivity.** The first request after idle pays a cold
start *and* a cold BM25 index — the index is built per tenant and policy scope on first
use (67.8ms, against 4.9ms warm). Both are visible in the dashboard's latency figure, and
neither is a bug.

## 3. Dashboard

A Vercel project with root directory `apps/web` and one variable:

```
NEXT_PUBLIC_API_BASE_URL=https://<your-api>.onrender.com
```

A trailing slash here is harmless now — the client strips it — but it was not always:
`https://host/` plus `/policies` gives `https://host//policies`, which FastAPI does not
route, so the dashboard showed "Cannot load the library — Not Found" while the API was
perfectly healthy.

It is `NEXT_PUBLIC_`, so it is compiled into the browser bundle and visible to anyone who
looks. That is correct — it is a URL. No secret belongs in `apps/web`; the OpenAI key
lives only in the API's environment.

## 4. Check it

```bash
curl https://<api>/ready
curl https://<api>/policies
```

The first returns `{"status":"ready"}`; the second lists the seeded demo policy with
status `ready`.

Then in the browser: ask a question, confirm the citation panel shows page, section and
both scores; ask something the wording does not answer and confirm the abstention renders
with its reason.

**The check that matters is the one that cannot be done locally:** upload a policy,
**redeploy the service**, and confirm the document is still listed and still answers.
That is the failure this phase existed to remove — the filesystem does not survive, and
the documents now live somewhere that does.

## What is deliberately not here

- **No authentication.** The demo is open, with a per-IP window and a global daily cap
  instead. The isolation mechanism is real and tested; what is stubbed is the
  identification of the principal. See `docs/decisions.md`.
- **No real policy documents.** The seeded wording is synthetic. The LIC documents the
  benchmark uses are third-party publications: measuring against them locally is one
  thing, serving them from a public URL is another (`docs/sample-data.md`).
- **One worker by default.** `WEB_CONCURRENCY` raises it, at the cost of one BM25 index
  per worker. The daily cap is counted in the database precisely so that it still holds
  when that number is greater than one.
