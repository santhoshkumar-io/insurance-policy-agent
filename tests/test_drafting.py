"""Candidate drafting renders proposals, never labels."""

from __future__ import annotations

from rag.evaluation.drafting import to_yaml_stub

CANDIDATE = {
    "question": 'Does the "room rent" cap apply per day?',
    "answer_in_source": "1% of sum insured per day",
    "why_hard": "the figure is in a table cell, not prose",
    "expected_pages": [4],
    "expected_section": "6. Table of Benefits",
    "chunk_type": "table",
    "source_chunk": "abc",
}


def test_stub_marks_every_candidate_unverified():
    """The marker is the safeguard: nothing unchecked may reach a published metric."""
    stub = to_yaml_stub([CANDIDATE, CANDIDATE, CANDIDATE], "acme-health-guard")
    notes = [ln for ln in stub.splitlines() if ln.strip().startswith("notes:")]
    assert len(notes) == 3
    assert all("UNVERIFIED" in note for note in notes)
    assert "verified by nobody" in stub


def test_stub_quotes_questions_so_punctuation_cannot_break_the_yaml():
    stub = to_yaml_stub([CANDIDATE], "acme-health-guard")
    assert 'question: "Does the \\"room rent\\" cap apply per day?"' in stub


def test_stub_carries_the_page_label_and_section():
    stub = to_yaml_stub([CANDIDATE], "acme-health-guard")
    assert "expected_pages: [4]" in stub
    assert 'expected_section: "6. Table of Benefits"' in stub
    assert "gold_answer: null" in stub


def test_stub_renders_a_missing_section_as_null_not_empty_string():
    candidate = {**CANDIDATE, "expected_section": None}
    assert "expected_section: null" in to_yaml_stub([candidate], "p")


def test_stub_includes_the_source_quote_as_a_comment_for_the_verifier():
    stub = to_yaml_stub([CANDIDATE], "acme-health-guard")
    assert "# source says: 1% of sum insured per day" in stub


def test_stub_ids_do_not_collide_with_the_verified_set():
    """Candidates get c-prefixed ids so they can never be mistaken for real questions."""
    stub = to_yaml_stub([CANDIDATE, CANDIDATE], "p", start_index=7)
    assert "id: c007" in stub and "id: c008" in stub
    assert "id: q" not in stub


def test_stub_of_nothing_is_still_a_warning_not_an_empty_file():
    stub = to_yaml_stub([], "p")
    assert "NOT YET A BENCHMARK" in stub


def test_drafted_yaml_is_parseable_and_rejected_by_the_loader_until_edited():
    """A candidate stub must not silently load as a benchmark.

    It parses as YAML, but it has no `version`/`set_name`, so load_eval_set cannot
    consume it by accident.
    """
    import yaml

    parsed = yaml.safe_load("questions:\n" + to_yaml_stub([CANDIDATE], "p"))
    assert isinstance(parsed["questions"], list)
    assert "version" not in parsed
