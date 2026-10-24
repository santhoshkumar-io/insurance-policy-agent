# Sample data provenance

The brief forbids real customer data in portfolio demos, and is right to. This file
records where every sample document came from, so the demo stays defensible.

| File | Insurer | Product | Source URL | Retrieved | Redistributable |
|---|---|---|---|---|---|
| _(none yet)_ | | | | | |

## Rules

1. Public policy wordings only. Insurers publish filed wordings as public PDFs
   (IRDAI-filed wordings in India; state-filed policy forms in the US).
2. Record the URL and retrieval date in the table above before ingesting.
3. **Check redistribution terms before committing a PDF to the repo.** `samples/*.pdf`
   is gitignored by default; prefer a download script over vendored binaries.
4. No customer names, policy numbers or PII, including in screenshots and the demo video.
