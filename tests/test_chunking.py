"""Chunking is the highest technical risk in this system, so it gets real assertions."""

from __future__ import annotations

from rag.chunking.chunker import chunk_document
from rag.ingestion.pdf import extract_document


def test_extracts_every_page_with_correct_numbering(structured_policy_pdf):
    doc = extract_document(structured_policy_pdf)
    assert doc.page_count == 4
    assert [p.number for p in doc.pages] == [1, 2, 3, 4]
    assert all(p.char_count > 0 for p in doc.pages)


def test_running_footer_is_stripped(structured_policy_pdf):
    doc = extract_document(structured_policy_pdf)
    text = " ".join(s.text for p in doc.pages for s in p.spans)
    # The footer appears on all three pages at the same y-position.
    assert "UIN ACM0001" not in text


def test_headings_become_section_paths(structured_policy_pdf):
    chunks = chunk_document(extract_document(structured_policy_pdf))
    paths = [c.section_path for c in chunks if c.section_path]
    assert paths, "no section paths detected on a document with clear headings"
    assert any("Waiting Periods" in p for p in paths)
    # Nesting: a sub-clause records its parent, not just itself.
    assert any(">" in p for p in paths)


def test_waiting_period_clause_lands_on_its_real_page(structured_policy_pdf):
    """The citation promise is only worth anything if page attribution is right."""
    chunks = chunk_document(extract_document(structured_policy_pdf))
    matches = [c for c in chunks if "thirty six months" in c.text]
    assert matches, "pre-existing disease waiting period clause was lost in chunking"
    assert all(c.page_from == 2 for c in matches)


def test_every_chunk_carries_the_metadata_contract(structured_policy_pdf):
    chunks = chunk_document(extract_document(structured_policy_pdf))
    assert chunks
    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.page_from >= 1
        assert chunk.page_to >= chunk.page_from
        assert chunk.chunk_type in {"prose", "table", "heading"}
        assert chunk.token_count > 0
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_unstructured_document_falls_back_to_windows(flat_policy_pdf):
    """No headings must not mean no chunks."""
    doc = extract_document(flat_policy_pdf)
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    assert all(c.section_path == "" for c in chunks)
    assert {c.page_from for c in chunks} & {1, 2, 3}


def test_chunks_stay_within_the_token_budget(structured_policy_pdf, flat_policy_pdf):
    for pdf in (structured_policy_pdf, flat_policy_pdf):
        for chunk in chunk_document(extract_document(pdf)):
            if chunk.chunk_type == "prose":
                assert chunk.token_count <= 900, f"runaway chunk: {chunk.token_count} tokens"


def test_benefits_table_becomes_its_own_chunk_with_the_grid_intact(structured_policy_pdf):
    """The limits agents ask about are tabular; a flattened table loses the evidence."""
    chunks = chunk_document(extract_document(structured_policy_pdf))
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert tables, "the ruled benefits table was not detected"

    grid = next((t for t in tables if "Cataract" in t.text), None)
    assert grid is not None, "benefits table content was lost"

    # Markdown rows keep each benefit with its own limit and waiting period.
    row = next(line for line in grid.text.splitlines() if "Cataract" in line)
    assert "40,000" in row
    assert "24 months" in row
    assert grid.page_from == grid.page_to == 4


def test_table_cells_keep_their_word_boundaries(structured_policy_pdf):
    """Glyph-positioned PDFs return cell text with the spaces missing unless rebuilt."""
    chunks = chunk_document(extract_document(structured_policy_pdf))
    grid = next(c for c in chunks if c.chunk_type == "table" and "Cataract" in c.text)
    assert "Room rent" in grid.text
    assert "sum insured per day" in grid.text
    assert "Roomrent" not in grid.text
