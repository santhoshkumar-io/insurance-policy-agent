"""The ceilings that make an open demo affordable.

The deployed instance has no authentication by design, so these limits are the only thing
between a stranger's loop and a personal OpenAI bill. They are tested at the unit level
rather than by hammering the API, because the behaviour that matters -- what happens on
request 11 -- is cheaper to assert directly than to reproduce.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException, Request
from sqlalchemy import func, select

from db.models import Conversation, Message, Tenant
from rag.config import get_settings
from services.api import limits


def _request(ip: str = "203.0.113.7", forwarded: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/chat",
            "headers": headers,
            "client": (ip, 51234),
            "server": ("testserver", 80),
        }
    )


@pytest.fixture(autouse=True)
def clean_window():
    limits.reset()
    yield
    limits.reset()


def test_requests_under_the_limit_pass():
    for _ in range(5):
        limits.enforce_ip_rate_limit(_request(), limit_per_minute=5)


def test_the_request_over_the_limit_is_refused_with_retry_after():
    for _ in range(3):
        limits.enforce_ip_rate_limit(_request(), limit_per_minute=3)

    with pytest.raises(HTTPException) as exc:
        limits.enforce_ip_rate_limit(_request(), limit_per_minute=3)

    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers


def test_one_caller_cannot_exhaust_another_callers_budget():
    """The window is per IP. A shared counter would let one loop lock everyone out."""
    for _ in range(3):
        limits.enforce_ip_rate_limit(_request(ip="198.51.100.1"), limit_per_minute=3)

    limits.enforce_ip_rate_limit(_request(ip="198.51.100.2"), limit_per_minute=3)


def test_the_forwarded_header_identifies_the_caller_behind_a_proxy():
    """Render terminates TLS in front of the app, so request.client is the proxy.

    Without this the whole internet shares one bucket and the first busy visitor locks
    out the rest.
    """
    for _ in range(3):
        limits.enforce_ip_rate_limit(
            _request(ip="10.0.0.1", forwarded="203.0.113.9"), limit_per_minute=3
        )

    # A different real caller arriving through the same proxy still has its own budget.
    limits.enforce_ip_rate_limit(
        _request(ip="10.0.0.1", forwarded="203.0.113.10"), limit_per_minute=3
    )


def test_a_limit_of_zero_disables_the_check():
    for _ in range(50):
        limits.enforce_ip_rate_limit(_request(), limit_per_minute=0)


def _asked_today(session) -> int:
    cutoff = datetime.now(UTC) - timedelta(days=1)
    return session.scalar(select(func.count(Message.id)).where(Message.created_at > cutoff)) or 0


def _conversation(session):
    """A tenant and conversation to hang test messages off."""
    tenant = Tenant(slug=f"t-{uuid.uuid4().hex[:8]}", name="Cap Tenant")
    session.add(tenant)
    session.flush()
    conversation = Conversation(tenant_id=tenant.id)
    session.add(conversation)
    session.flush()
    return tenant, conversation


@pytest.mark.db
def test_the_daily_cap_counts_questions_across_tenants(session, monkeypatch):
    """The bill is global, so the cap is global -- and it is counted from the database.

    An in-process counter would reset on every restart and be per worker, so N workers
    would allow N times the intended spend.

    The cap is set relative to what the database already holds: a development database
    has a day's evaluation runs in it, and an absolute number would make this test pass
    or fail depending on what else ran today.
    """
    settings = get_settings()
    tenant, conversation = _conversation(session)
    monkeypatch.setattr(settings, "daily_question_cap", _asked_today(session) + 2)

    limits.enforce_daily_cap(session)  # under the cap

    for _ in range(2):
        session.add(
            Message(
                tenant_id=tenant.id,
                conversation_id=conversation.id,
                question="q",
                answer="a",
            )
        )
    session.flush()

    with pytest.raises(HTTPException) as exc:
        limits.enforce_daily_cap(session)
    assert exc.value.status_code == 429
    assert "daily quota" in exc.value.detail


@pytest.mark.db
def test_yesterdays_questions_do_not_count_against_today(session, monkeypatch):
    """The window slides. Without that, the cap is a lifetime limit."""
    settings = get_settings()
    tenant, conversation = _conversation(session)
    monkeypatch.setattr(settings, "daily_question_cap", _asked_today(session) + 1)

    session.add(
        Message(
            tenant_id=tenant.id,
            conversation_id=conversation.id,
            question="q",
            answer="a",
            created_at=datetime.now(UTC) - timedelta(days=2),
        )
    )
    session.flush()

    limits.enforce_daily_cap(session)
