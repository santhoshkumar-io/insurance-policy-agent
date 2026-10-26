from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from db.session import get_session_factory
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
    """Is the process alive? Deliberately shallow, and deliberately dependency-free.

    A liveness probe that touches the database restarts a healthy API every time the
    database hiccups, which turns a brief outage into a crash loop.
    """
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready() -> dict:
    """Can this instance actually serve a request?

    This is what a load balancer should route on. Without it a container reports green
    while every query fails -- the API is up, the database is unreachable, and the only
    symptom is 500s. ``SELECT 1`` also proves the connection pool can hand out a
    connection, not merely that a host name resolves.
    """
    try:
        with get_session_factory()() as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(503, f"database unavailable: {type(exc).__name__}") from exc
    return {"status": "ready"}
