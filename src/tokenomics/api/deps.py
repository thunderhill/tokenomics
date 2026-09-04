"""Shared dependencies: settings, database handles, filters, and optional auth."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Annotated

import psycopg
from fastapi import Depends, HTTPException, Query, Request, status

from tokenomics.pricing.engine import PricingEngine, default_engine
from tokenomics.settings import Settings
from tokenomics.storage import database
from tokenomics.storage.queries import DIMENSIONS, Filters

DEFAULT_WINDOW_DAYS = 30


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def get_engine() -> PricingEngine:
    """The vendored pricing snapshot, loaded once per process."""
    return default_engine()


def get_db() -> Iterator[psycopg.Connection]:
    """A pooled connection. The pool commits on success and rolls back on error."""
    with database.connection() as conn:
        yield conn


def require_api_key(request: Request) -> None:
    """Bearer-token check, active only when TOKENOMICS_API_KEY is set."""
    import hmac

    expected = get_settings().api_key
    if not expected:
        return
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_filters(
    since: Annotated[datetime | None, Query(description="Inclusive lower bound (UTC).")] = None,
    until: Annotated[datetime | None, Query(description="Exclusive upper bound (UTC).")] = None,
    project: Annotated[list[str] | None, Query()] = None,
    feature: Annotated[list[str] | None, Query()] = None,
    environment: Annotated[list[str] | None, Query()] = None,
    subject_id: Annotated[list[str] | None, Query()] = None,
    prompt_version: Annotated[list[str] | None, Query()] = None,
    model_key: Annotated[list[str] | None, Query()] = None,
    provider: Annotated[list[str] | None, Query()] = None,
    tag: Annotated[
        list[str] | None, Query(description="Repeatable ``key=value`` tag filter.")
    ] = None,
) -> Filters:
    """Build a query filter from repeatable query parameters.

    Every value lands in a bound parameter; nothing here is interpolated into SQL.
    """
    now = datetime.now(UTC)
    tags: dict[str, str] = {}
    for item in tag or []:
        key, separator, value = item.partition("=")
        if not separator:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"tag filter {item!r} must be key=value",
            )
        tags[key] = value

    return Filters(
        since=_aware(since) or now - timedelta(days=DEFAULT_WINDOW_DAYS),
        until=_aware(until) or now,
        project=tuple(project or ()),
        feature=tuple(feature or ()),
        environment=tuple(environment or ()),
        subject_id=tuple(subject_id or ()),
        prompt_version=tuple(prompt_version or ()),
        model_key=tuple(model_key or ()),
        provider=tuple(provider or ()),
        tags=tags,
    )


def _aware(moment: datetime | None) -> datetime | None:
    if moment is None:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def parse_group_by(group_by: list[str] | None) -> list[str]:
    """Split repeated or comma-separated dimensions and check them against the whitelist."""
    dimensions: list[str] = []
    for item in group_by or []:
        dimensions.extend(part.strip() for part in item.split(",") if part.strip())
    unknown = [d for d in dimensions if d not in DIMENSIONS]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"unknown dimension(s) {unknown}; allowed: {sorted(DIMENSIONS)}",
        )
    return dimensions


Db = Annotated[psycopg.Connection, Depends(get_db)]
Engine = Annotated[PricingEngine, Depends(get_engine)]
Config = Annotated[Settings, Depends(get_settings)]
FilterQuery = Annotated[Filters, Depends(get_filters)]
