"""Chunking against a document shaped like a real IRDAI-filed wording.

Everything here is a regression test for something that was actually broken. The chunker
passed every test against the clean synthetic fixture while producing section paths like
`` >  > PART- A`` and ``18. Endorsement means conditions attached/ affixed to this
Policy...`` on real LIC documents.

The real PDFs cannot be committed, so `build_lic_shaped_policy` reproduces their
structural traps: headings at body point size, a PART backbone, numbered definition
entries, and a bare page number that differs on every page.
"""

from __future__ import annotations

from rag.chunking.chunker import (
    DEFINITION_SENTENCE,
    NUMERIC_ONLY,
    PART_HEADING,
    chunk_document,
)
from rag.ingestion.pdf import extract_document

# ------------------------------------------------------------------ pattern unit tests


def test_part_heading_matches_the_dash_variants_real_documents_use():
    """The LIC wordings mix hyphen, en dash and spacing inconsistently across parts."""
    for text in (
        "PART- A",
        "PART– B: DEFINITIONS",
        "PART — C: BENEFITS",
        "PART – D: CONDITIONS RELATED TO SERVICING ASPECTS",
        "PART E",
        "PART - G: STATUTORY PROVISIONS",
    ):
        assert PART_HEADING.match(text), text


def test_part_heading_rejects_the_lower_case_cross_reference():
    """"...are set out in Part C of this Policy Document." is a sentence, not a heading.

    It appears in both real documents and was being matched as a top-level PART, which
    would reset the section backbone mid-sentence.
    """
    assert not PART_HEADING.match("Part C of this Policy Document.")
    assert not PART_HEADING.match("part b")


def test_part_heading_does_not_match_an_unrelated_word():
    assert not PART_HEADING.match("PARTIAL WITHDRAWAL")
    assert not PART_HEADING.match("PARTICULARS OF THE PROPOSER")


def test_definition_sentence_pattern_catches_the_real_offenders():
    for text in (
        "18. Endorsement means conditions attached/ affixed to this Policy",
        "28. Material information shall mean all important, essential and relevant",
        "35. Policyholder is the legal owner of this policy.",
        "40. Proposer is a person who proposes the life insurance proposal.",
    ):
        assert DEFINITION_SENTENCE.match(text), text


def test_definition_pattern_leaves_real_numbered_headings_alone():
    for text in ("1. Proof of Age:", "4. Surrender:", "7. Grace period:", "5. Policy Loan:"):
        assert not DEFINITION_SENTENCE.match(text), text


def test_numeric_only_matches_page_furniture():
    for text in ("1", "12", "- 3 -", "4."):
        assert NUMERIC_ONLY.match(text), text
    assert not NUMERIC_ONLY.match("1. Proof of Age:")


# ------------------------------------------------------------------- document assertions


def test_every_section_sits_under_its_part(lic_shaped_pdf):
    """The PART backbone must never be displaced.

    A numbered clause claiming level 1 wiped out the parent for every chunk that
    followed, which is how "1. Proof of Age:" lost "PART – D".
    """
    chunks = chunk_document(extract_document(lic_shaped_pdf))
    paths = [c.section_path for c in chunks if c.section_path]
    assert paths
    assert all(p.startswith("PART") for p in paths), [p for p in paths if not p.startswith("PART")]


def test_numbered_clause_headings_keep_their_part_parent(lic_shaped_pdf):
    chunks = chunk_document(extract_document(lic_shaped_pdf))
    grace = next(c for c in chunks if "grace period of thirty days" in c.text)
    assert "PART - D" in grace.section_path
    assert "7. Grace period:" in grace.section_path


def test_section_paths_have_no_empty_levels(lic_shaped_pdf):
    """Padding levels rendered as " >  > PART- A" on every real document."""
    chunks = chunk_document(extract_document(lic_shaped_pdf))
    for chunk in chunks:
        assert " >  > " not in chunk.section_path
        assert not chunk.section_path.startswith(" >")
        assert not chunk.section_path.endswith("> ")


def test_page_numbers_never_become_headings(lic_shaped_pdf):
    """A bare footer digit differs on every page, so identical-text stripping missed it.

    17 of 71 accepted headings in one real document were page numbers.
    """
    chunks = chunk_document(extract_document(lic_shaped_pdf))
    for chunk in chunks:
        for level in chunk.section_path.split(" > "):
            assert not level.strip().isdigit(), chunk.section_path


def test_page_number_text_is_stripped_from_the_body(lic_shaped_pdf):
    doc = extract_document(lic_shaped_pdf)
    for page in doc.pages:
        assert not any(s.text.strip() == str(page.number) for s in page.spans)


def test_numbered_definitions_stay_body_text(lic_shaped_pdf):
    """They are entries in PART B, not headings -- however emphatically typeset."""
    chunks = chunk_document(extract_document(lic_shaped_pdf))
    definition = next(c for c in chunks if "Endorsement means" in c.text)
    assert definition.section_path == "PART- B: DEFINITIONS"
    assert all("Endorsement means" not in c.section_path for c in chunks)
    assert all("Material information" not in c.section_path for c in chunks)


def test_bold_lead_in_does_not_promote_a_sentence_to_a_heading(lic_shaped_pdf):
    """`_group_lines` took the max span size, so a bold "18." inflated the whole line."""
    chunks = chunk_document(extract_document(lic_shaped_pdf))
    for chunk in chunks:
        for level in chunk.section_path.split(" > "):
            assert len(level) < 70, level


def test_headings_are_found_despite_sharing_the_body_font_size(lic_shaped_pdf):
    """The whole difficulty of these documents: prominence alone cannot find headings."""
    doc = extract_document(lic_shaped_pdf)
    chunks = chunk_document(doc)
    paths = {c.section_path for c in chunks}
    assert any("PART- A" in p for p in paths)
    assert any("PART- B" in p for p in paths)
    assert any("PART - D" in p for p in paths)
    assert any("PART - G" in p for p in paths)


def test_literal_tokens_survive_into_a_chunk(lic_shaped_pdf):
    """UINs and statutory references are what the keyword arm exists to find."""
    chunks = chunk_document(extract_document(lic_shaped_pdf))
    text = " ".join(c.text for c in chunks)
    assert "512N356V02" in text
    assert "Section 45" in text


def test_running_header_is_kept_once_and_stripped_thereafter(lic_shaped_pdf):
    """Furniture detection must not delete substantive text outright.

    Found by a benchmark question: "What is the UIN of this plan?" was unanswerable
    because the UIN sits in the running header of every page, so the corpus contained it
    zero times. The insurer name and plan name have the same problem -- genuinely
    repeated furniture and genuinely substantive. Keeping the first occurrence preserves
    the information once while still removing the noise.
    """
    doc = extract_document(lic_shaped_pdf)
    corpus = " ".join(s.text for page in doc.pages for s in page.spans)
    assert "LIFE ASSURANCE CORPORATION" in corpus
    assert corpus.count("LIFE ASSURANCE CORPORATION") == 1


def test_the_first_occurrence_kept_is_the_earliest_page(lic_shaped_pdf):
    doc = extract_document(lic_shaped_pdf)
    pages_with_title = [
        page.number
        for page in doc.pages
        if any("LIFE ASSURANCE CORPORATION" in s.text for s in page.spans)
    ]
    assert pages_with_title == [1]
