"""Runtime configuration. Every value has a working default so the API boots offline."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from tokenomics.storage.database import DEFAULT_DSN


class Settings(BaseSettings):
    """Read from ``TOKENOMICS_*`` environment variables or a ``.env`` file."""

    model_config = SettingsConfigDict(env_prefix="TOKENOMICS_", env_file=".env", extra="ignore")

    database_url: str = DEFAULT_DSN

    #: Run migrations and create the current month's partition at startup. Convenient
    #: for self-hosting; turn it off if your database user is not the schema owner.
    auto_migrate: bool = True

    #: Dashboard origins allowed to call the API from a browser.
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://localhost:4173")

    #: Reject OTLP bodies larger than this before decoding them.
    max_body_bytes: int = 8 * 1024 * 1024

    #: When set, every /v1 and /api request must carry ``Authorization: Bearer <key>``.
    #: Unset by default so a local ``docker compose up`` just works.
    api_key: str | None = None

    #: Refresh hourly rollups for this many hours behind each ingest batch.
    rollup_window_hours: int = Field(default=3, ge=1)


def load_settings() -> Settings:
    return Settings()
