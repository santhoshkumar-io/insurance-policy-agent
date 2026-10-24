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

    retrieval_top_k: int = 8
    abstain_score_threshold: float = 0.25

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
