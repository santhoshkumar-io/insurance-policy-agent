"""PDF extraction that preserves the structure chunking needs.

Plain ``page.get_text()`` throws away the two things this system depends on: the font
metrics that reveal headings, and the row/column relationships in tables. Since the
answers to the questions agents actually ask -- sub-limits, waiting periods, room-rent
caps -- live in tables, flattening them loses the evidence (plan.md §2.1).

So this module extracts spans with their font metrics and tables separately, and leaves
the interpretation to :mod:`rag.chunking`.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # PyMuPDF < 1.24.3
    import fitz


@dataclass
class Span:
    """A run of text with the font metrics that let us guess at structure."""

    text: str
    size: float
    bold: bool
    page: int  # 1-based, matching what a human sees in a PDF reader
    y0: float
    x0: float


@dataclass
class Table:
    page: int
    markdown: str
    rows: int
    cols: int


@dataclass
class Page:
    number: int
    spans: list[Span] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.spans)


@dataclass
class ExtractedDocument:
    path: Path
    page_count: int
    pages: list[Page]
    checksum: str
    # Pages whose extracted text is so thin they are probably scans.
    low_text_pages: list[int] = field(default_factory=list)

    @property
    def body_font_size(self) -> float:
        """The modal font size, used as the baseline that headings stand out from."""
        sizes = Counter(
            round(s.size, 1) for p in self.pages for s in p.spans if len(s.text.strip()) > 20
        )
        if not sizes:
            return 10.0
        return sizes.most_common(1)[0][0]


def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _cell_text(words: list, bbox) -> str:
    """Rebuild a cell's text from the words whose centre falls inside it.

    ``Table.extract()`` returns cell text straight from the content stream, which in
    PDFs that position each glyph individually comes back with the spaces missing --
    "AIEngineering/DocumentIntelligence/SaaS". The word list splits on spatial gaps as
    well as space characters, so reconstructing from it recovers the boundaries.
    """
    if bbox is None:
        return ""
    x0, y0, x1, y1 = bbox
    inside = [
        w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1
    ]
    # Band the y-coordinate before sorting: glyphs from a different font on the same
    # visual line (arrows, symbols) sit at a slightly different baseline and would
    # otherwise sort into their own row and land at the end of the cell.
    inside.sort(key=lambda w: (round(w[1] / 3.0), w[0]))
    return _normalise(" ".join(w[4] for w in inside))


def _extract_tables(page) -> list[Table]:
    """Serialise each detected table as markdown so the grid survives into the prompt."""
    tables: list[Table] = []
    try:
        found = page.find_tables()
    except Exception:
        return tables

    words = page.get_text("words")

    for table in getattr(found, "tables", []):
        data: list[list[str]] = []
        try:
            for row in getattr(table, "rows", []):
                data.append([_cell_text(words, bbox) for bbox in row.cells])
        except Exception:
            data = []

        if not any(any(c for c in row) for row in data):
            # Fall back to the library's own extraction rather than dropping the table.
            try:
                data = [[c or "" for c in row] for row in table.extract()]
            except Exception:
                continue

        rows = [[_normalise(c or "") for c in row] for row in data if any(c for c in row)]
        if len(rows) < 2:
            continue  # a single row is not a table worth isolating
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        header, *body = rows
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        lines += ["| " + " | ".join(r) + " |" for r in body]
        tables.append(
            Table(page=page.number + 1, markdown="\n".join(lines), rows=len(rows), cols=width)
        )
    return tables


def _strip_running_headers(pages: list[Page], min_fraction: float = 0.6) -> None:
    """Drop text that repeats at the same place on most pages.

    Page furniture pollutes embeddings and keyword search alike, and it is easy to spot:
    identical text recurring at a near-identical y-position across most of the document.

    The ``max(3, ...)`` floor below means a three-page document only loses text that
    appears on all three pages -- short documents get the conservative treatment, since
    a repeated line there is more likely to be real content.
    """
    if len(pages) < 3:
        return

    def band(y: float) -> int:
        return int(y // 12)

    counts: Counter[tuple[str, int]] = Counter()
    for page in pages:
        seen = {(_normalise(s.text), band(s.y0)) for s in page.spans if s.text.strip()}
        counts.update(seen)

    threshold = max(3, int(len(pages) * min_fraction))
    furniture = {key for key, n in counts.items() if n >= threshold and len(key[0]) < 120}
    if not furniture:
        return

    for page in pages:
        page.spans = [
            s for s in page.spans if (_normalise(s.text), band(s.y0)) not in furniture
        ]


def extract_document(path: str | Path, max_pages: int = 600) -> ExtractedDocument:
    path = Path(path)
    doc = fitz.open(path)
    try:
        if doc.page_count > max_pages:
            raise ValueError(
                f"{path.name} has {doc.page_count} pages, above the {max_pages}-page limit"
            )

        pages: list[Page] = []
        for index in range(doc.page_count):
            fitz_page = doc[index]
            page = Page(number=index + 1)

            blocks = fitz_page.get_text("dict").get("blocks", [])
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if not text.strip():
                            continue
                        flags = span.get("flags", 0)
                        bbox = span.get("bbox", (0, 0, 0, 0))
                        page.spans.append(
                            Span(
                                text=text,
                                size=float(span.get("size", 0.0)),
                                bold=bool(flags & 2 ** 4),  # PyMuPDF bold flag
                                page=index + 1,
                                y0=float(bbox[1]),
                                x0=float(bbox[0]),
                            )
                        )

            page.tables = _extract_tables(fitz_page)
            pages.append(page)

        _strip_running_headers(pages)

        return ExtractedDocument(
            path=path,
            page_count=doc.page_count,
            pages=pages,
            checksum=file_checksum(path),
            low_text_pages=[p.number for p in pages if p.char_count < 50],
        )
    finally:
        doc.close()
