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


@dataclass
class _Section:
    path: list[str] = field(default_factory=list)

    def as_str(self) -> str:
        return " > ".join(self.path)


def _group_lines(spans: list[Span], y_tolerance: float = 2.5) -> list[_Line]:
    """Reassemble spans into visual lines. Spans are fragments; lines are meaning."""
    lines: list[_Line] = []
    for span in sorted(spans, key=lambda s: (s.page, round(s.y0, 1), s.x0)):
        if lines:
            last = lines[-1]
            if last.page == span.page and abs(last.y0 - span.y0) <= y_tolerance:
                joiner = "" if last.text.endswith(("-", " ")) else " "
                last.text = (last.text + joiner + span.text).strip()
                last.size = max(last.size, span.size)
                last.bold = last.bold or span.bold
                continue
        lines.append(
            _Line(
                text=span.text.strip(),
                size=span.size,
                bold=span.bold,
                page=span.page,
                y0=span.y0,
                x0=span.x0,
            )
        )
    return [ln for ln in lines if ln.text]


def _heading_level(line: _Line, body_size: float) -> int | None:
    """Return a heading depth (1 = top level), or None if this is body text.

    Two independent signals, because policy PDFs use either or both: visual prominence
    (larger or bold) and clause numbering depth.
    """
    text = line.text.strip()
    if not text or len(text) > 140:
        return None

    words = text.split()
    # A heading names a section; it does not make a statement. These three shapes are
    # all body text that font metrics alone would misread as a heading -- most often a
    # bulleted list whose lead-in phrase is bold.
    if text.endswith((".", ";")) and len(words) > 6:
        return None
    if ":" in text and len(text.split(":", 1)[1].strip()) > 30:
        return None
    if len(words) > 14:
        return None

    match = CLAUSE_NUMBER.match(text)
    numbered = bool(match)
    prominent = line.size >= body_size * 1.08 or (line.bold and line.size >= body_size * 0.98)

    if numbered and (prominent or len(text) < 90):
        number = match.group(1).rstrip(".")
        if number.startswith("("):
            return 3
        return min(1 + number.count("."), 4)

    if prominent and len(text) < 90:
        if line.size >= body_size * 1.35:
            return 1
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
        for line in _group_lines(page.spans):
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


def chunk_document(doc: ExtractedDocument) -> list[ChunkDraft]:
    body_size = doc.body_font_size
    all_lines = [ln for page in doc.pages for ln in _group_lines(page.spans)]

    headings = sum(1 for ln in all_lines if _heading_level(ln, body_size) is not None)
    structured = headings >= MIN_HEADINGS_FOR_STRUCTURED

    drafts: list[ChunkDraft] = []

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
                    section_path="",
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
