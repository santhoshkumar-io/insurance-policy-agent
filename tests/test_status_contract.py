"""The frontend's idea of a document status must match the backend's.

`apps/web/src/lib/api.ts` mirrors the Python schemas by hand, which is a duplication the
module's own docstring calls out as a drift risk. It then drifted: the frontend declared
a status called "processing" that the backend never emits, and watched for it to decide
whether ingestion was still running. The result was an upload screen that stopped polling
the instant the document moved from "pending" to "parsing" and sat there claiming nothing
had happened, while ingestion completed normally a few seconds later.

Nothing caught it. Both sides compiled, every Python test passed, the browser smoke test
checked that the file input was *visible* and never submitted one. So this test reads the
TypeScript and compares it to the enum, because a comment warning about drift does not
prevent drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from db.models import DocumentStatus

API_TS = Path("apps/web/src/lib/api.ts")
HOOK_TS = Path("apps/web/src/hooks/use-policies.ts")

# Statuses that mean ingestion is still running, so the dashboard must keep polling.
IN_FLIGHT = {DocumentStatus.PENDING, DocumentStatus.PARSING, DocumentStatus.EMBEDDING}
SETTLED = {DocumentStatus.READY, DocumentStatus.FAILED}


def _declared_statuses() -> set[str]:
    source = API_TS.read_text()
    match = re.search(r"export type DocumentStatus\s*=\s*([^;]+);", source)
    assert match, "DocumentStatus union not found in api.ts"
    return set(re.findall(r'"([a-z_]+)"', match.group(1)))


def _busy_statuses() -> set[str]:
    source = HOOK_TS.read_text()
    match = re.search(r"const BUSY[^=]*=\s*new Set\(\[([^\]]*)\]\)", source)
    assert match, "BUSY set not found in use-policies.ts"
    return set(re.findall(r'"([a-z_]+)"', match.group(1)))


def test_the_frontend_declares_exactly_the_backend_statuses():
    assert _declared_statuses() == set(DocumentStatus.ALL)


def test_every_in_flight_status_keeps_the_dashboard_polling():
    """The actual bug: a missing in-flight status silently ends the poll loop."""
    missing = IN_FLIGHT - _busy_statuses()
    assert not missing, (
        f"{sorted(missing)} mean ingestion is still running, but the dashboard does not "
        f"treat them as busy -- so it stops polling while the work is in flight."
    )


def test_no_settled_status_is_treated_as_busy():
    """The opposite failure: polling forever on a document that has finished."""
    assert not (_busy_statuses() & SETTLED)


@pytest.mark.parametrize("status", sorted(DocumentStatus.ALL))
def test_every_status_has_a_badge(status):
    """A status with no badge variant renders with fallback styling and no meaning."""
    source = Path("apps/web/src/components/status-badge.tsx").read_text()
    assert re.search(rf"\b{status}:", source), f"{status} has no entry in the badge map"
