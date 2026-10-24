"""Engine, session factory, and the tenant scoping helper.

Every read of tenant-owned data goes through :func:`scoped`. That is the whole point:
one place to audit instead of every query in the codebase (plan.md §2.5).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Select, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from db.models import Tenant
from rag.config import get_settings

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(get_settings().database_url, pool_pre_ping=True, future=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def scoped(stmt: Select, model, tenant_id: uuid.UUID) -> Select:
    """Constrain a SELECT to one tenant.

    Applied in SQL, never by filtering in Python after the fact -- a Python-side filter
    still pulls another tenant's rows across the boundary before discarding them.
    """
    if not hasattr(model, "tenant_id"):
        raise TypeError(f"{model.__name__} is not tenant-scoped; it must not hold tenant data")
    return stmt.where(model.tenant_id == tenant_id)


def get_or_create_dev_tenant(session: Session) -> Tenant:
    """Week 1 stands in for auth with a fixed tenant.

    Note what this does *not* do: it does not bypass scoping. The isolation boundary is
    live from day 1; only the authentication in front of it is stubbed.
    """
    slug = get_settings().dev_tenant_slug
    tenant = session.scalar(select(Tenant).where(Tenant.slug == slug))
    if tenant is None:
        tenant = Tenant(slug=slug, name="Development Tenant")
        session.add(tenant)
        session.flush()
    return tenant
