"""Request dependencies.

Week 1 stands in for authentication with a fixed development tenant. Note what is *not*
stubbed: the tenant boundary itself. Every query still goes through the scoping helper,
and the isolation test runs in CI from day 1. Only the identification of the principal
is temporary (plan.md §2.5).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

from sqlalchemy.orm import Session

from db.session import get_or_create_dev_tenant, get_session_factory


def get_db() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def current_tenant_id(session: Session) -> uuid.UUID:
    tenant = get_or_create_dev_tenant(session)
    session.commit()
    return tenant.id
