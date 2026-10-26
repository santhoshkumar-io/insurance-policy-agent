"""The browser-facing CORS policy.

The dashboard runs on another origin, so the API needs CORS to be callable at all. These
tests exist because the easy way to make that work is ``allow_origins=["*"]``, which then
ships to deployment because nobody revisits it. The wildcard assertion is the point of the
module; the preflight test only proves the middleware is wired at all.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag.config import get_settings
from services.api.main import app

client = TestClient(app)


def test_a_configured_origin_is_allowed_through_preflight():
    origin = get_settings().cors_origins[0]
    response = client.options(
        "/policies",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_an_unlisted_origin_is_refused():
    response = client.options(
        "/policies",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_the_allowed_origins_are_not_a_wildcard():
    """A wildcard would let any page in any tab call this API on a reader's behalf."""
    assert "*" not in get_settings().cors_origins


def test_credentials_are_not_allowed():
    """The API carries no cookie session, so nothing needs credentialed requests.

    Allowing them would mean a browser attaches cookies to cross-origin calls, which is
    the part of CORS that turns a permissive origin list into a real vulnerability.
    """
    origin = get_settings().cors_origins[0]
    response = client.options(
        "/policies",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-credentials" not in response.headers


# --------------------------------------------------------------------------------------
# Parsing the setting. This value is typed by hand into a hosting dashboard's environment
# form, and the settings object is built at import time -- so a value the parser rejects
# does not misconfigure CORS, it stops the API booting. These cases are the ones a person
# actually types.
# --------------------------------------------------------------------------------------

import pytest  # noqa: E402

from rag.config import Settings  # noqa: E402

VERCEL = "https://insurance-policy-agent.vercel.app"


@pytest.mark.parametrize(
    "raw",
    [
        f'["{VERCEL}"]',  # the documented JSON form
        f"['{VERCEL}']",  # Python syntax, which is what a developer types
        VERCEL,  # the bare URL, which is what everyone else types
        f'"{VERCEL}"',  # quoted, from a copy out of a JSON file
    ],
)
def test_every_plausible_way_of_writing_one_origin_works(monkeypatch, raw):
    monkeypatch.setenv("CORS_ORIGINS", raw)
    assert Settings(_env_file=None).cors_origins == [VERCEL]


def test_a_comma_separated_list_is_accepted(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", f"{VERCEL}, http://localhost:3000")
    assert Settings(_env_file=None).cors_origins == [VERCEL, "http://localhost:3000"]


def test_a_trailing_slash_is_stripped(monkeypatch):
    """The hardest failure to see: a browser's Origin header never has a trailing slash.

    `https://app.vercel.app/` in the allowlist parses fine, looks right in the dashboard,
    and matches nothing -- every request fails CORS with no error on the server side.
    """
    monkeypatch.setenv("CORS_ORIGINS", f'["{VERCEL}/"]')
    assert Settings(_env_file=None).cors_origins == [VERCEL]


def test_an_empty_value_allows_nothing_rather_than_everything(monkeypatch):
    """Failing closed. An empty list blocks browsers; it does not become a wildcard."""
    monkeypatch.setenv("CORS_ORIGINS", "")
    assert Settings(_env_file=None).cors_origins == []
