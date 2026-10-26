# Dashboard

Next.js + TypeScript + Tailwind + shadcn/ui. Three screens: **upload**, **library**, **ask**.

Its job is to make the retrieval engineering visible, not to be a product surface. The
citation panel is the reason it exists: for every chunk the model was served it shows the
page and section path **read from the database row** (never from model output), which arm
retrieved it, the text itself, and **both scores side by side** — the ranking score, whose
scale depends on the retrieval mode, next to the cosine, which does not. Those two being
one unlabelled number was a real bug in this codebase; see `docs/decisions.md`.

Abstentions render as abstentions, with their reason. A blank answer box would undo the
feature the system is built around.

```bash
uvicorn services.api.main:app --reload   # from the repository root
npm install
npm run dev                              # http://localhost:3000
```

`NEXT_PUBLIC_API_BASE_URL` points at the API (see `.env.example`). It is public by
necessity — the browser calls the API directly — so no secret belongs in it. The API key
lives in the backend's `.env` only.

## Checks

```bash
npm run lint
npx next typegen && npx tsc --noEmit   # typegen writes the generated route types first
npm run smoke                          # drives all three screens in a real browser
```

`smoke` needs a running API, a database with an ingested policy, and an OpenAI key,
because the two answers it checks are a cited answer and an abstention. It is not in CI
for that reason — but it caught two defects the build was happy with. Set
`SMOKE_SHOT_DIR` to also write full-page screenshots.
