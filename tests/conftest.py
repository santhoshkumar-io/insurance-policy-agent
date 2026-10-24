"""Test fixtures.

The policy PDFs are generated rather than vendored (see ``scripts/sample_policy.py``):
a fixture whose exact layout the tests know is the only way to assert that a heading
became a section path and that page 2 really is page 2.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.sample_policy import (
    build_flat_policy,
    build_injected_policy,
    build_structured_policy,
)

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")


@pytest.fixture(scope="session")
def structured_policy_pdf(tmp_path_factory) -> Path:
    """A policy with clear headings, numbered clauses and a benefits table."""
    return build_structured_policy(tmp_path_factory.mktemp("pdfs") / "structured_policy.pdf")


@pytest.fixture(scope="session")
def flat_policy_pdf(tmp_path_factory) -> Path:
    """A policy with no structure at all -- must trigger the windowed fallback."""
    return build_flat_policy(tmp_path_factory.mktemp("pdfs") / "flat_policy.pdf")


@pytest.fixture(scope="session")
def injected_policy_pdf(tmp_path_factory) -> Path:
    """Red-team fixture: a policy carrying instructions aimed at the model."""
    return build_injected_policy(tmp_path_factory.mktemp("pdfs") / "injected_policy.pdf")
