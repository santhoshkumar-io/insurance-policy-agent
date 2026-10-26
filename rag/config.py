"""Single source of configuration. Model IDs are pinned here, not scattered in call sites."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536

    database_url: str = "postgresql+psycopg://policy:policy@localhost:5433/policy_intelligence"

    max_upload_bytes: int = 50 * 1024 * 1024
    max_pages: int = 600
    ocr_min_chars_per_page: int = 50

    # Set from the benchmark. On the real LIC corpus hybrid RRF(vector + BM25) retrieves
    # the evidence for *every* answerable question (Recall@5 = 1.000) where vector alone
    # reaches 0.949, and the cached BM25 index made it deployable at ~5ms warm. Reranking
    # has a better MRR (0.932) but costs 5 points of recall and 10x the latency, and
    # missing evidence is unrecoverable while imperfect ordering inside the served set is
    # not -- the model sees all top_k chunks either way. See docs/decisions.md.
    retrieval_mode: str = "hybrid_bm25"  # vector | keyword | bm25 | hybrid | hybrid_bm25
    retrieval_top_k: int = 8
    retrieval_candidates: int = 40  # per arm, before fusion
    rrf_k: int = 60

    # Off until the benchmark justifies the latency and the torch dependency.
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # How many scoped BM25 indexes to keep in memory. Each worker holds its own, so this
    # multiplies by worker count -- see rag/retrieval/bm25.py on what that costs.
    bm25_cache_size: int = 8

    # Section metadata as a tie-breaker rather than a retrieval arm. The arm version was
    # measured and lost; this is the retry. Off until it earns its place.
    # The tolerance is relative, so one value works across cosine, RRF and logit scales.
    section_tiebreak_enabled: bool = False
    section_tiebreak_tolerance: float = 0.15

    # Compared against RetrievedChunk.confidence -- a cosine similarity -- never against
    # the active strategy's ranking score. The old name (abstain_score_threshold) is what
    # invited the bug: an RRF score of 0.016 fails a 0.25 threshold every time, so
    # enabling hybrid retrieval silently made the system refuse every question.
    abstain_semantic_threshold: float = 0.25

    dev_tenant_slug: str = "dev"

    # Ceilings for the open demo (services/api/limits.py). The per-IP window is held in
    # process and so is per worker; the daily cap is counted from the messages table and
    # so is not. Set either to 0 to disable it.
    rate_limit_questions_per_minute: int = 10
    rate_limit_uploads_per_minute: int = 3
    daily_question_cap: int = 300

    # Browser origins allowed to call the API. A list rather than "*" even in development:
    # the dashboard runs on a known port, and a wildcard that ships to deployment because
    # nobody revisited it is the usual way this setting becomes a vulnerability.
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    @field_validator("database_url")
    @classmethod
    def pin_the_driver(cls, value: str) -> str:
        """Accept a managed provider's connection string unmodified.

        Neon, Supabase and every other managed Postgres hand out a URL beginning
        ``postgresql://``, and the natural thing for an operator to do is paste it
        straight into the environment. Which driver that selects is decided by
        SQLAlchemy's default, and that default *changed*: 2.0 resolves a bare
        ``postgresql://`` to psycopg2, 2.1 resolves it to psycopg 3. Only psycopg 3 is
        installed here, so the same connection string works today and would fail with a
        ModuleNotFoundError on a 2.0 pin -- a failure whose message says nothing about
        the URL that caused it.

        Naming the driver removes the dependency on that default without asking the
        operator to hand-edit a secret, which is its own small source of outages.
        """
        for prefix in ("postgresql://", "postgres://"):
            if value.startswith(prefix):
                return "postgresql+psycopg://" + value[len(prefix) :]
        return value

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.openai_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
