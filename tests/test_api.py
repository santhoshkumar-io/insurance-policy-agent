"""API surface tests. The upload validator runs without a database."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.api.main import app
from services.api.routes.policies import _validate_upload


def test_health_is_shallow():
    """Liveness must not touch the database, or a database blip becomes a crash loop."""
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.db
def test_ready_reports_the_database():
    with TestClient(app) as client:
        assert client.get("/ready").json() == {"status": "ready"}


def test_ready_fails_when_the_database_is_unreachable(monkeypatch):
    """The point of a separate probe: green must mean *able to serve*.

    Without this check an instance with no database advertises itself as healthy and
    answers every request with a 500.
    """

    def broken_factory():
        raise RuntimeError("connection refused")

    monkeypatch.setattr("services.api.main.get_session_factory", broken_factory)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert "database unavailable" in response.json()["detail"]


def test_upload_rejects_a_non_pdf_whatever_the_extension_says():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _validate_upload(b"<?php system($_GET['c']); ?>", "policy.pdf")
    assert exc.value.status_code == 415


def test_upload_rejects_an_empty_file():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _validate_upload(b"", "policy.pdf")
    assert exc.value.status_code == 400


def test_upload_rejects_an_oversized_file(monkeypatch):
    from fastapi import HTTPException

    from rag.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "max_upload_bytes", 10)
    with pytest.raises(HTTPException) as exc:
        _validate_upload(b"%PDF-" + b"x" * 100, "policy.pdf")
    assert exc.value.status_code == 413


def test_upload_accepts_a_real_pdf(structured_policy_pdf):
    _validate_upload(structured_policy_pdf.read_bytes(), "policy.pdf")
