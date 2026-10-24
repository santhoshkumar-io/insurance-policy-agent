"""Cross-encoder reranking.

A bi-encoder scores query and chunk independently; a cross-encoder reads them together
and is markedly better at judging relevance. It is also slower, and it drags torch into
the image.

So this is off by default and must earn its place on the benchmark. The brief's own stack
principle: "use established components and measure their value." If the measured gain
does not justify the latency and image size, the honest outcome is a row in
docs/experiments.md saying so -- not a reranker nobody evaluated.
"""

from __future__ import annotations

from functools import lru_cache

from rag.config import get_settings
from rag.retrieval.types import RetrievedChunk

INSTALL_HINT = (
    "Reranking needs the optional extra: `uv pip install -e \".[rerank]\"` "
    "(adds sentence-transformers and torch)."
)


@lru_cache
def _model(name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(INSTALL_HINT) from exc
    return CrossEncoder(name)


def rerank(
    question: str, chunks: list[RetrievedChunk], top_k: int | None = None
) -> list[RetrievedChunk]:
    """Reorder by cross-encoder relevance.

    ``semantic_score`` is left untouched. The cross-encoder logit is unbounded and often
    negative, so it replaces ``score`` only -- the confidence signal abstention reads
    stays on the cosine scale it was calibrated against.
    """
    if not chunks:
        return []

    settings = get_settings()
    model = _model(settings.rerank_model)
    scores = model.predict([(question, chunk.text) for chunk in chunks])

    ordered = sorted(
        zip(chunks, scores, strict=True), key=lambda pair: float(pair[1]), reverse=True
    )
    result: list[RetrievedChunk] = []
    for position, (chunk, score) in enumerate(ordered, start=1):
        chunk.score = float(score)
        chunk.rank = position
        result.append(chunk)

    return result[: top_k or settings.retrieval_top_k]
