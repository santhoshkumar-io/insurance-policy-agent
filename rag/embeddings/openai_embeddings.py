"""OpenAI embeddings, batched.

Every vector is written alongside the model that produced it. Mixing vectors from two
models in one similarity search is a silent correctness failure, not an error (plan.md §2.9).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

from rag.config import get_settings

BATCH_SIZE = 96


@lru_cache
def get_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.require_openai_key())


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    dim: int


def embed_texts(texts: list[str]) -> EmbeddingResult:
    settings = get_settings()
    client = get_client()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = [t.replace("\x00", " ") for t in texts[start : start + BATCH_SIZE]]
        response = client.embeddings.create(model=settings.openai_embedding_model, input=batch)
        vectors.extend(item.embedding for item in response.data)

    dim = len(vectors[0]) if vectors else settings.openai_embedding_dim
    if vectors and dim != settings.openai_embedding_dim:
        raise RuntimeError(
            f"Embedding model returned {dim} dimensions but the schema expects "
            f"{settings.openai_embedding_dim}. Changing the model requires a migration "
            f"and a full re-index."
        )
    return EmbeddingResult(vectors=vectors, model=settings.openai_embedding_model, dim=dim)


def embed_query(text: str) -> list[float]:
    return embed_texts([text]).vectors[0]
