"""Rate limits for an open demo.

The deployed instance has no authentication by design, which means an unmetered path to a
paid OpenAI key. Two ceilings, because they fail differently:

* **Per IP, per minute** — stops one person or one loop from monopolising the service.
  Held in process, so each worker enforces its own share. That is approximate, and it is
  the right kind of approximate: the cost of being wrong is that a burst gets through.

* **Globally, per day** — the one that protects the bill. Counted from the ``messages``
  table rather than from memory, because an in-process counter resets on every restart
  and is per worker, so N workers would allow N times the intended spend. The table is
  shared state that already exists and already records exactly the thing being capped.

Neither needs Redis, and adding it would be a third service to deploy for a demo whose
whole point is that it is cheap to run.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Message
from rag.config import get_settings

# ip -> timestamps of recent requests. Unbounded in principle; bounded in practice by
# pruning on every lookup, which also keeps the dict from holding yesterday's visitors.
_HITS: dict[str, deque[float]] = defaultdict(deque)


def reset() -> None:
    """Clear the in-process window. For tests, which must not inherit each other's state."""
    _HITS.clear()


def _client_ip(request: Request) -> str:
    """The caller's address, trusting one hop of proxy.

    Render and Vercel both terminate TLS in front of the app, so ``request.client.host``
    is the proxy. ``X-Forwarded-For`` is caller-controlled and therefore spoofable -- this
    limit is a cost control, not a security boundary, and the daily cap below is what
    holds when someone decides to rotate the header.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_ip_rate_limit(request: Request, limit_per_minute: int) -> None:
    if limit_per_minute <= 0:
        return

    now = time.monotonic()
    hits = _HITS[_client_ip(request)]
    while hits and now - hits[0] > 60.0:
        hits.popleft()

    if len(hits) >= limit_per_minute:
        retry_after = int(60.0 - (now - hits[0])) + 1
        raise HTTPException(
            429,
            detail=(
                f"Rate limit: {limit_per_minute} requests a minute. "
                f"Try again in {retry_after}s."
            ),
            headers={"Retry-After": str(retry_after)},
        )
    hits.append(now)


def enforce_daily_cap(session: Session) -> None:
    """Stop the day's spend at a fixed number of answered questions.

    Deliberately *not* scoped to a tenant: the bill is global, so the cap is global. This
    is the one count in the codebase that crosses the tenant boundary, and it crosses it
    without reading any tenant's data -- it counts rows and returns an integer.
    """
    settings = get_settings()
    if settings.daily_question_cap <= 0:
        return

    cutoff = datetime.now(UTC) - timedelta(days=1)
    asked = session.scalar(select(func.count(Message.id)).where(Message.created_at > cutoff))
    if (asked or 0) >= settings.daily_question_cap:
        raise HTTPException(
            429,
            detail=(
                "This demo has answered its daily quota of questions. It runs on a "
                "personal API key, so the cap is the point. Try again tomorrow, or run "
                "it locally -- the repository has the instructions."
            ),
        )
