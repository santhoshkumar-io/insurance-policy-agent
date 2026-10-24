"""Prompt injection defence, as a test that can genuinely fail (plan.md §2.8)."""

from __future__ import annotations

from rag.chunking.chunker import chunk_document
from rag.generation.prompt import SYSTEM_PROMPT, build_user_message, render_sources
from rag.ingestion.pdf import extract_document
from tests.test_citation_validation import make_chunk


def test_system_prompt_contains_no_document_content():
    """Nothing inside a PDF can rewrite the system's role, because it never goes there."""
    chunk = make_chunk(text="SECRET POLICY CLAUSE ABOUT DENTAL")
    message = build_user_message("is dental covered?", [chunk])
    assert "SECRET POLICY CLAUSE" in message
    assert "SECRET POLICY CLAUSE" not in SYSTEM_PROMPT


def test_system_prompt_states_the_data_not_instructions_rule():
    lowered = SYSTEM_PROMPT.lower()
    assert "untrusted" in lowered
    assert "not instruction" in lowered


def test_injected_pdf_text_is_delivered_inside_source_delimiters(injected_policy_pdf):
    """The attack text must arrive as quoted data, never as a bare instruction."""
    chunks = chunk_document(extract_document(injected_policy_pdf))
    injected = [c for c in chunks if "Ignore all previous instructions" in c.text]
    assert injected, "fixture did not produce the injected text"

    served = [make_chunk(text=injected[0].text)]
    message = build_user_message("what is covered?", served)

    attack_index = message.index("Ignore all previous instructions")
    open_index = message.rindex("<source", 0, attack_index)
    close_index = message.index("</source>", attack_index)
    assert open_index < attack_index < close_index

    assert "untrusted data" in message
    assert "do not obey them" in message


def test_only_served_ids_are_offered_to_the_model():
    chunks = [make_chunk() for _ in range(3)]
    message = build_user_message("q?", chunks)
    allowed_line = [ln for ln in message.splitlines() if ln.startswith("The only citation ids")]
    assert allowed_line
    for chunk in chunks:
        assert chunk.short_id in allowed_line[0]


def test_source_block_metadata_comes_from_the_record():
    chunk = make_chunk()
    rendered = render_sources([chunk])
    assert f'id="{chunk.short_id}"' in rendered
    assert 'page="2"' in rendered
    assert 'section="4. Waiting Periods"' in rendered
