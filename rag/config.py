"""Single source of configuration. Model IDs are pinned here, not scattered in call sites."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536

    database_url: str = "postgresql+psycopg://policy:policy@localhost:5433/policy_intelligence"
    storage_dir: Path = Path("./storage")

    max_upload_bytes: int = 50 * 1024 * 1024
    max_pages: int = 600
    ocr_min_chars_per_page: int = 50

    # Default set from the benchmark, not from preference: hybrid retrieval measured
    # *worse* than vector alone on the current corpus (docs/experiments.md). Hybrid,
    # BM25 and the reranker all stay built and are one setting away.
    retrieval_mode: str = "vector"  # vector | keyword | bm25 | hybrid | hybrid_bm25
    retrieval_top_k: int = 8
    retrieval_candidates: int = 40  # per arm, before fusion
    rrf_k: int = 60

    # Off until the benchmark justifies the latency and the torch dependency.
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Compared against RetrievedChunk.confidence -- a cosine similarity -- never against
    # the active strategy's ranking score. The old name (abstain_score_threshold) is what
    # invited the bug: an RRF score of 0.016 fails a 0.25 threshold every time, so
    # enabling hybrid retrieval silently made the system refuse every question.
    abstain_semantic_threshold: float = 0.25

    dev_tenant_slug: str = "dev"

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.openai_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
