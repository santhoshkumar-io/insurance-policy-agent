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
