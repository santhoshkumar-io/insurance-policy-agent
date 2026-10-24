"""Shared retrieval types.

``RetrievedChunk`` lives here rather than inside one strategy because generation,
validation, the API, the CLI and the evaluation harness all speak in terms of it. Keeping
it next to the vector strategy made ``from rag.retrieval.vector import RetrievedChunk``
read as though vector search owned the concept.

The two score fields are the important part of this file.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum


class RetrievalMode(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    # In-memory BM25. Not a shipping mode -- it exists so the claim "BM25 beats
    # ts_rank_cd here" can be measured rather than asserted (plan.md §2.3).
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_BM25 = "hybrid_bm25"

    @property
    def uses_vector(self) -> bool:
        return self in {
            RetrievalMode.VECTOR,
            RetrievalMode.HYBRID,
            RetrievalMode.HYBRID_BM25,
        }


class RetrievedBy(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    BOTH = "both"


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    text: str
    page_from: int
    page_to: int
    section_path: str
    chunk_type: str
    policy_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    policy_name: str

    # The score the active strategy ranked by. Its scale depends on the strategy:
    # cosine similarity, ts_rank_cd, an RRF sum or a cross-encoder logit. Useful for
    # ordering and for display; meaningless to compare against a fixed threshold.
    score: float

    # Cosine similarity, always populated whatever the ranking mode -- including when a
    # chunk was found only by the keyword arm, in which case it is computed for exactly
    # this purpose. Abstention reads *this*, so its threshold stays comparable across
    # every retrieval experiment. See plan.md and docs/decisions.md on the score-scale
    # trap: an abstention threshold tuned for cosine similarity silently rejects every
    # answer once ranking switches to RRF.
    semantic_score: float | None = None

    rank: int = 0
    retrieved_by: RetrievedBy = RetrievedBy.VECTOR

    @property
    def short_id(self) -> str:
        """The handle the model is allowed to cite. Short enough not to waste tokens."""
        return f"c_{str(self.chunk_id)[:8]}"

    @property
    def pages_label(self) -> str:
        return (
            str(self.page_from)
            if self.page_from == self.page_to
            else f"{self.page_from}-{self.page_to}"
        )

    def covers_page(self, page: int) -> bool:
        """Whether this chunk's page range contains ``page``.

        Evaluation labels evidence by page rather than by chunk id, so that re-chunking
        does not invalidate the benchmark. This is how a label is matched.
        """
        return self.page_from <= page <= self.page_to

    @property
    def confidence(self) -> float:
        """The score abstention and threshold calibration use."""
        return self.semantic_score if self.semantic_score is not None else self.score
