"""Structure-aware chunking for insurance policy wordings.

Strategy, in order of preference (plan.md §2.1):

1. Detect headings from font metrics relative to the document's modal body font, and
   from clause numbering (``4.2.1``), maintaining a running section path so every chunk
   knows its address in the document.
2. Emit each table as its own chunk, keeping the markdown grid intact, because the
   limits and waiting periods agents ask about are tabular and do not survive
   linearisation.
3. Fall back to windowed chunking with overlap whenever heading detection finds too
   little -- many policy PDFs are flat exports or scans with no usable structure, and
   the pipeline must not require structure to exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.chunking.tokens import count_tokens
from rag.ingestion.pdf import ExtractedDocument, Span

TARGET_TOKENS = 450
MAX_TOKENS = 700
OVERLAP_TOKENS = 60
MIN_CHUNK_TOKENS = 25
MIN_HEADINGS_FOR_STRUCTURED = 3

# "4.", "4.2", "4.2.1", "(a)", "IV." -- clause numbering as policies actually write it.
CLAUSE_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)*\.?|\([a-z]\)|\([ivxl]+\)|[IVXL]+\.)\s+\S")
SENTENCE_END = re.compile(r"(?<=[.;:])\s+")

# "PART- A", "PART– B: DEFINITIONS", "PART – D: CONDITIONS...". Every IRDAI-filed policy
# wording is organised into these parts, and in the LIC documents they are the only
# reliable structural signal -- headings there are the same point size as body text, so
# visual prominence cannot find them. Uppercase is required: the lower-case
# cross-reference "Part C of this Policy Document." is a sentence, not a heading.
PART_HEADING = re.compile(r"^PART\s*[-–—]?\s*([A-G])\b(?!\w)")

# "18. Endorsement means ...", "28. Material information shall mean ...". These are
# definition entries in PART B -- body text that happens to start with a clause number.
DEFINITION_SENTENCE = re.compile(
    r"^\s*\d+\.\s+.{0,60}?\b(means|shall mean|is|are|refers to|includes)\b", re.I
)

# A line that is only digits (optionally with punctuation) is page furniture, never a
# heading. Page numbers were being promoted to level-3 headings on every page.
NUMERIC_ONLY = re.compile(r"^[\d\s.,:;|/\-–—()]+$")


@dataclass
class ChunkDraft:
    text: str
    page_from: int
    page_to: int
    section_path: str
    chunk_type: str
    ordinal: int = 0
    token_count: int = 0


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    page: int
    y0: float
    x0: float
    # Character-weighted accumulators, so `size` and `bold` describe what the line mostly
    # *is* rather than what its most prominent fragment is. Taking the max let a bold "18."
    # marker promote a whole definition sentence to a top-level heading.
    _weights: dict = field(default_factory=dict)
    _bold_chars: int = 0
    _total_chars: int = 0

    def absorb(self, text: str, size: float, bold: bool) -> None:
        n = max(1, len(text.strip()))
        key = round(size, 1)
        self._weights[key] = self._weights.get(key, 0) + n
        self._total_chars += n
        if bold:
            self._bold_chars += n
        self.size = max(self._weights, key=lambda k: self._weights[k])
        # Bold only when bold dominates the line, not when any fragment is bold.
        self.bold = self._bold_chars * 2 > self._total_chars


@dataclass
class _Section:
    path: list[str] = field(default_factory=list)

    def as_str(self) -> str:
        # Drop padding levels. _push_section pads to reach a depth, which otherwise
        # renders as " >  > PART- A".
        return " > ".join(part for part in self.path if part.strip())


def _group_lines(spans: list[Span], y_tolerance: float = 2.5) -> list[_Line]:
    """Reassemble spans into visual lines. Spans are fragments; lines are meaning."""
    lines: list[_Line] = []
    for span in sorted(spans, key=lambda s: (s.page, round(s.y0, 1), s.x0)):
        if lines:
            last = lines[-1]
            if last.page == span.page and abs(last.y0 - span.y0) <= y_tolerance:
                joiner = "" if last.text.endswith(("-", " ")) else " "
                last.text = (last.text + joiner + span.text).strip()
                last.absorb(span.text, span.size, span.bold)
                continue
        line = _Line(
            text=span.text.strip(),
            size=span.size,
            bold=span.bold,
            page=span.page,
            y0=span.y0,
            x0=span.x0,
        )
        line.absorb(span.text, span.size, span.bold)
        lines.append(line)
    return [ln for ln in lines if ln.text]


def _heading_level(line: _Line, body_size: float) -> int | None:
    """Return a heading depth (1 = top level), or None if this is body text.

    Two independent signals, because policy PDFs use either or both: visual prominence
    (larger or bold) and clause numbering depth.
    """
    text = line.text.strip()
    if not text or len(text) > 140:
        return None

    # The PART backbone first: it is the one signal that survives documents where
    # headings share the body font size.
    if PART_HEADING.match(text) and len(text) < 80 and not text.endswith("."):
        return 1

    # Page numbers and other numeric furniture.
    if NUMERIC_ONLY.match(text):
        return None

    # Quote marks mean quoted material, not a section name. Catches line-grouping
    # artifacts like `Death ”.` where a fragment of the following line merged in.
    if any(ch in text for ch in '"\u201c\u201d'):
        return None

    # A numbered definition is body text, however emphatically it is typeset.
    if DEFINITION_SENTENCE.match(text):
        return None

    words = text.split()
    # A heading names a section; it does not make a statement. These three shapes are
    # all body text that font metrics alone would misread as a heading -- most often a
    # bulleted list whose lead-in phrase is bold.
    if text.endswith((".", ";")) and len(words) > 6:
        return None
    if ":" in text and len(text.split(":", 1)[1].strip()) > 30:
        return None
    # A heading names a section; past about ten words it is making a statement instead.
    if len(words) > 10:
        return None

    match = CLAUSE_NUMBER.match(text)
    numbered = bool(match)
    prominent = line.size >= body_size * 1.08 or (line.bold and line.size >= body_size * 0.98)

    # Level 1 belongs to PART alone. A numbered clause or a bold schedule label that
    # claimed it would wipe out the document's structural backbone for every chunk that
    # followed -- which is how "1. Proof of Age:" lost its PART D parent and a stray
    # "Name" on the schedule page displaced PART A.
    if numbered and (prominent or len(text) < 90):
        number = match.group(1).rstrip(".")
        if number.startswith("("):
            return 4
        return min(2 + number.count("."), 4)

    if prominent and len(text) < 90:
        if line.size >= body_size * 1.15:
            return 2
        return 3

    return None


def _push_section(section: _Section, level: int, title: str) -> None:
    del section.path[level - 1 :]
    while len(section.path) < level - 1:
        section.path.append("")
    section.path.append(title)


def _split_oversized(text: str, limit: int) -> list[str]:
    """Split on sentence boundaries, not mid-clause, when a section exceeds the budget."""
    if count_tokens(text) <= limit:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    for sentence in SENTENCE_END.split(text):
        candidate = " ".join([*current, sentence]).strip()
        if current and count_tokens(candidate) > limit:
            parts.append(" ".join(current).strip())
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        parts.append(" ".join(current).strip())
    return [p for p in parts if p]


def _window_fallback(doc: ExtractedDocument) -> list[ChunkDraft]:
    """Overlapping windows over raw page text, used when there is no usable structure."""
    drafts: list[ChunkDraft] = []
    buffer: list[str] = []
    buffer_pages: list[int] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(buffer).strip()
        if text:
            drafts.append(
                ChunkDraft(
                    text=text,
                    page_from=min(buffer_pages),
                    page_to=max(buffer_pages),
                    section_path="",
                    chunk_type="prose",
                    token_count=count_tokens(text),
                )
            )

    for page in doc.pages:
        for line in _outside_tables(page, _group_lines(page.spans)):
            buffer.append(line.text)
            buffer_pages.append(line.page)
            if count_tokens(" ".join(buffer)) >= TARGET_TOKENS:
                flush()
                # Carry an overlap so a clause split across a boundary stays retrievable.
                tail: list[str] = []
                tail_pages: list[int] = []
                for text, pg in zip(reversed(buffer), reversed(buffer_pages), strict=True):
                    tail.insert(0, text)
                    tail_pages.insert(0, pg)
                    if count_tokens(" ".join(tail)) >= OVERLAP_TOKENS:
                        break
                buffer, buffer_pages = tail, tail_pages
    flush()
    return drafts


def _outside_tables(page, lines: list[_Line]) -> list[_Line]:
    """Drop lines that fall inside a detected table.

    Their content is already carried by the table chunk, as a markdown grid with the
    row and column relationships intact. Leaving them in the prose stream indexes every
    cell twice and lets a bold header row ("Benefit | Limit | Waiting period") pass for
    a section heading on font metrics alone.
    """
    if not page.tables:
        return lines
    return [
        ln for ln in lines
        if not any(table.contains(ln.x0, ln.y0) for table in page.tables)
    ]


def chunk_document(doc: ExtractedDocument) -> list[ChunkDraft]:
    body_size = doc.body_font_size
    all_lines = [
        ln for page in doc.pages for ln in _outside_tables(page, _group_lines(page.spans))
    ]

    headings = sum(1 for ln in all_lines if _heading_level(ln, body_size) is not None)
    structured = headings >= MIN_HEADINGS_FOR_STRUCTURED

    drafts: list[ChunkDraft] = []

    # (page, y, section_path) for every heading, in document order. A table is emitted
    # in a separate pass, so this is how it recovers the section it sits under -- a
    # citation reading "6. Table of Benefits" tells the agent far more than a blank one.
    heading_marks: list[tuple[int, float, str]] = []

    if structured:
        section = _Section()
        buffer: list[str] = []
        pages: list[int] = []
        buffer_section = ""

        def flush() -> None:
            nonlocal buffer, pages
            if not buffer:
                return
            text = " ".join(buffer).strip()
            if text:
                for part in _split_oversized(text, MAX_TOKENS):
                    drafts.append(
                        ChunkDraft(
                            text=part,
                            page_from=min(pages),
                            page_to=max(pages),
                            section_path=buffer_section,
                            chunk_type="prose",
                            token_count=count_tokens(part),
                        )
                    )
            buffer, pages = [], []

        for line in all_lines:
            level = _heading_level(line, body_size)
            if level is not None:
                flush()
                _push_section(section, level, line.text.strip())
                buffer_section = section.as_str()
                heading_marks.append((line.page, line.y0, buffer_section))
                continue
            if not buffer:
                buffer_section = section.as_str()
            buffer.append(line.text)
            pages.append(line.page)
            if count_tokens(" ".join(buffer)) >= TARGET_TOKENS:
                flush()
        flush()
    else:
        drafts = _window_fallback(doc)

    # Tables are emitted separately in page order so the grid reaches the model intact.
    for page in doc.pages:
        for table in page.tables:
            drafts.append(
                ChunkDraft(
                    text=table.markdown,
                    page_from=table.page,
                    page_to=table.page,
                    section_path=_section_for(heading_marks, table.page, table.bbox[1]),
                    chunk_type="table",
                    token_count=count_tokens(table.markdown),
                )
            )

    drafts = _merge_runts(drafts)

    drafts.sort(key=lambda d: (d.page_from, d.chunk_type == "table"))
    for i, draft in enumerate(drafts):
        draft.ordinal = i
    return drafts


def _merge_runts(drafts: list[ChunkDraft]) -> list[ChunkDraft]:
    """Fold tiny prose chunks into their neighbour.

    A heading immediately followed by another heading leaves a few-token fragment
    behind. Those pollute the index and make for a citation that shows the agent
    nothing, so they are merged rather than stored.
    """
    merged: list[ChunkDraft] = []
    for draft in drafts:
        if (
            draft.chunk_type == "prose"
            and draft.token_count < MIN_CHUNK_TOKENS
            and merged
            and merged[-1].chunk_type == "prose"
            and merged[-1].token_count + draft.token_count <= MAX_TOKENS
        ):
            previous = merged[-1]
            previous.text = f"{previous.text} {draft.text}".strip()
            previous.page_to = max(previous.page_to, draft.page_to)
            previous.token_count = count_tokens(previous.text)
            continue
        merged.append(draft)
    return merged


def _section_for(
    marks: list[tuple[int, float, str]], page: int, y: float
) -> str:
    """The section path in force at a position: the last heading at or before it."""
    current = ""
    for mark_page, mark_y, path in marks:
        if (mark_page, mark_y) <= (page, y):
            current = path
        else:
            break
    return current
