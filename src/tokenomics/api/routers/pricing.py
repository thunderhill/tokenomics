"""Pricing snapshot inspection and refresh.

Refreshing is the **only** outbound network call this service ever makes, and it is
always explicit. Everything else runs against the snapshot vendored in the package, so
an air-gapped deployment is fully functional.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from tokenomics.api.deps import Db, Engine
from tokenomics.models import Component, ServiceTier
from tokenomics.pricing.grammar import CACHE_TTL_DEFAULT
from tokenomics.pricing.pricebook import LITELLM_URL, PriceBook
from tokenomics.storage import repository
from tokenomics.telemetry import metrics

router = APIRouter(prefix="/api/pricing", tags=["pricing"])


class SnapshotOut(BaseModel):
    snapshot_id: str
    source: str
    fetched_at: datetime
    model_count: int


class ModelRatesOut(BaseModel):
    model_key: str
    provider: str | None
    mode: str | None
    max_input_tokens: int | None
    context_tiers: list[int]
    prices_cache: bool
    input_usd_per_token: Decimal | None
    output_usd_per_token: Decimal | None
    cache_read_usd_per_token: Decimal | None
    cache_write_usd_per_token: Decimal | None


@router.get("/snapshot", response_model=SnapshotOut, summary="The active pricing snapshot")
def snapshot(engine: Engine) -> SnapshotOut:
    book = engine.book
    metrics.pricebook_models.labels(snapshot=book.snapshot_id).set(len(book))
    return SnapshotOut(
        snapshot_id=book.snapshot_id,
        source=book.source,
        fetched_at=book.fetched_at,
        model_count=len(book),
    )


@router.get("/models", response_model=list[str], summary="Search model keys")
def models(
    engine: Engine,
    q: Annotated[str, Query(min_length=1, description="Substring match on the key.")],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[str]:
    needle = q.lower()
    return sorted(key for key in engine.book.models if needle in key.lower())[:limit]


@router.get("/models/{model_key:path}", response_model=ModelRatesOut, summary="Rates for a model")
def model_rates(engine: Engine, model_key: str) -> ModelRatesOut:
    resolution = engine.resolver.resolve(model_key)
    pricing = resolution.pricing
    if pricing is None or resolution.model_key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no pricing for {model_key!r}")

    def rate(component: Component) -> Decimal | None:
        return pricing.rate(
            component,
            context_tier=None,
            service_tier=ServiceTier.STANDARD,
            cache_ttl=CACHE_TTL_DEFAULT,
        )

    return ModelRatesOut(
        model_key=resolution.model_key,
        provider=pricing.provider,
        mode=pricing.mode,
        max_input_tokens=pricing.max_input_tokens,
        context_tiers=list(pricing.context_tiers),
        prices_cache=pricing.prices(Component.CACHE_READ),
        input_usd_per_token=rate(Component.INPUT),
        output_usd_per_token=rate(Component.OUTPUT),
        cache_read_usd_per_token=rate(Component.CACHE_READ),
        cache_write_usd_per_token=rate(Component.CACHE_WRITE),
    )


@router.post(
    "/refresh",
    response_model=SnapshotOut,
    summary="Fetch a fresh price list",
    description=(
        "Downloads the current LiteLLM price list and registers it as a new "
        "content-addressed snapshot. Already-priced events keep the snapshot and the "
        "exact rates they were billed with, so history never moves under your feet."
    ),
)
def refresh(conn: Db, url: str = LITELLM_URL) -> SnapshotOut:
    try:
        book = PriceBook.fetch(url)
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"could not fetch pricing: {exc}") from exc
    repository.record_snapshot(conn, book)
    return SnapshotOut(
        snapshot_id=book.snapshot_id,
        source=book.source,
        fetched_at=book.fetched_at,
        model_count=len(book),
    )
