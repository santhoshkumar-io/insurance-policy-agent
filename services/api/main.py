from __future__ import annotations

from fastapi import FastAPI

from services.api.routes import chat, policies

app = FastAPI(
    title="Insurance Policy Intelligence AI",
    version="0.1.0",
    description=(
        "Evidence-backed question answering over insurance policy documents. "
        "Answers are grounded in retrieved text with validated citations, and the "
        "system abstains rather than guessing. Not a claims authority."
    ),
)

app.include_router(policies.router)
app.include_router(chat.router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}
