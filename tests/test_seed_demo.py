"""Seeding the demo policy.

A deployment that starts with an empty library looks broken to the first person who opens
it, so the container seeds a synthetic policy on boot. Running on every boot means the
command has to be idempotent, and the first version was not: it keyed on the file's
checksum, and the generator writes a fresh creation timestamp into every PDF, so each
start added another copy. These tests pin both halves of that lesson.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import func, select

from db.models import Policy
from db.session import get_or_create_dev_tenant, scoped, session_scope
from rag.cli import DEMO_PRODUCT, seed_demo
from scripts.sample_policy import build_structured_policy


def test_the_generated_demo_pdf_is_not_byte_stable():
    """Why `seed-demo` cannot key on the checksum.

    If this ever starts failing, the generator has become deterministic and keying the
    seed on content would be legitimate -- but until then, two runs produce two different
    checksums for the same document, and an idempotence check built on them seeds a
    duplicate on every container start.
    """
    with tempfile.TemporaryDirectory() as workdir:
        first = build_structured_policy(Path(workdir) / "a.pdf").read_bytes()
        second = build_structured_policy(Path(workdir) / "b.pdf").read_bytes()

    assert hashlib.sha256(first).hexdigest() != hashlib.sha256(second).hexdigest()


@pytest.mark.db
def test_seeding_twice_leaves_one_demo_policy(stub_openai):
    """The boot path: whatever the starting state, a second run adds nothing."""
    seed_demo()
    seed_demo()

    with session_scope() as session:
        tenant = get_or_create_dev_tenant(session)
        count = session.scalar(
            scoped(
                select(func.count(Policy.id)).where(Policy.product_name == DEMO_PRODUCT),
                Policy,
                tenant.id,
            )
        )

    assert count == 1
