from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.config import get_settings
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

# The dashboard is served from another origin, so the browser needs this to call the API
# at all. Origins come from settings (CORS_ORIGINS) rather than being hardcoded to "*".
# Credentials stay off: the API carries no cookie session, and allowing both credentials
# and a wildcard is the combination browsers reject anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(policies.router)
app.include_router(chat.router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}
