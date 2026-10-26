"""The connection string a managed provider hands you must work unmodified.

Neon, Supabase and the rest emit `postgresql://...`. Which driver that selects is
SQLAlchemy's default, and that default changed between 2.0 (psycopg2) and 2.1 (psycopg
3) -- so a URL that works on one pin raises ModuleNotFoundError on the other, with a
message that never mentions the URL. Pinning the driver ourselves makes the paste-it-in
path work and removes the dependency on a default we do not control.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from rag.config import Settings

NEON = (
    "postgresql://owner:npg_secret@ep-sparkling-sun-123456-pooler"
    ".us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
)


def _settings(url: str) -> Settings:
    # _env_file=None so a developer's own .env cannot decide the outcome of this test.
    return Settings(database_url=url, _env_file=None)


@pytest.mark.parametrize("prefix", ["postgresql://", "postgres://"])
def test_a_providers_url_gets_the_driver_pinned(prefix):
    url = NEON.replace("postgresql://", prefix)
    assert _settings(url).database_url.startswith("postgresql+psycopg://")


def test_the_rest_of_the_url_survives_untouched():
    """Credentials, host, database and query parameters must pass through exactly.

    sslmode and channel_binding are not decoration: Neon refuses connections without
    TLS, so dropping the query string turns into a connection error far from here.
    """
    result = _settings(NEON).database_url
    assert result.endswith(
        "@ep-sparkling-sun-123456-pooler.us-east-2.aws.neon.tech"
        "/neondb?sslmode=require&channel_binding=require"
    )
    assert "owner:npg_secret" in result


def test_an_explicit_driver_is_left_alone():
    url = "postgresql+psycopg://user:pw@localhost:5433/policy_intelligence"
    assert _settings(url).database_url == url


def test_the_pinned_url_actually_selects_psycopg3():
    """The assertion that matters: this is the driver that is installed."""
    engine = create_engine(_settings(NEON).database_url)
    assert engine.dialect.driver == "psycopg"


def test_the_local_default_is_unchanged():
    """The development default already names its driver and must not be rewritten."""
    assert Settings(_env_file=None).database_url.startswith("postgresql+psycopg://")
