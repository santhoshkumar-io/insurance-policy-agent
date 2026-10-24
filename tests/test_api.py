"""API surface tests. The upload validator runs without a database."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.api.main import app
from services.api.routes.policies import _validate_upload


def test_health():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


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
