"""Attribution queries: spend sliced by any combination of tags.

Every aggregate carries both halves of the story -- the tokens and what they cost --
because one without the other is not actionable. Bear in mind while reading the
response shapes that ``input_tokens`` and ``output_tokens`` are the *reported*
totals, which already contain the cache and reasoning counts; the disjoint buckets
that were actually billed are in ``token_components``. See
:mod:`tokenomics.finops.tokens`.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from tokenomics.api.deps import Db, FilterQuery, parse_group_by
from tokenomics.finops import tokens as token_service
from tokenomics.storage import queries
from tokenomics.storage.queries import Granularity

router = APIRouter(prefix="/api/spend", tags=["spend"])


class CostComponents(BaseModel):
    """Spend split across the disjoint buckets it was actually billed from.

    These sum to ``cost_usd``. They are not the same thing as multiplying the reported
    token counts by a rate card: ``input_usd`` covers only the input that was *not*
    served from cache. See :mod:`tokenomics.pricing.cost`.
    """

    input_usd: Decimal = Decimal(0)
    output_usd: Decimal = Decimal(0)
    cache_read_usd: Decimal = Decimal(0)
    cache_write_usd: Decimal = Decimal(0)
    reasoning_usd: Decimal = Decimal(0)


class SpendRow(BaseModel):
    """One aggregated slice. Dimension values are echoed back in ``dimensions``."""

    dimensions: dict[str, str | None] = Field(default_factory=dict)
    requests: int
    subjects: int
    cost_usd: Decimal
    #: Reported totals, in semconv form: ``input_tokens`` already includes the cache
    #: counts and ``output_tokens`` already includes reasoning.
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    #: ``input_tokens`` minus the cached portion: what was billed at the full input
    #: rate. This is the figure that moves when a cache stops working.
    billable_input_tokens: int
    cost_components: CostComponents = Field(default_factory=CostComponents)
    #: Events in this slice we could not price. ``cost_usd`` excludes them, so a
    #: non-zero count here means the real figure is higher.
    unpriced_events: int


class SeriesPoint(SpendRow):
    bucket: datetime


class UnitEconomicsRow(SpendRow):
    cost_per_request: Decimal | None = None
    cost_per_subject: Decimal | None = None
    cost_per_1k_tokens: Decimal | None = None
    #: The blended effective rate, in the unit every provider quotes prices in.
    usd_per_1m_tokens: Decimal | None = None
    cache_hit_rate: float | None = None
    #: Share of output tokens spent on reasoning. ``None`` when there was no output.
    reasoning_share: float | None = None


class TokenEconomicsRow(SpendRow):
    """Token volume beside the cost of that volume, per disjoint billable component."""

    total_tokens: int
    token_components: dict[str, int]
    #: Each component's share of the billed token volume, and of the bill. The gap
    #: between the two is the point: output is routinely a tenth of the volume and
    #: half the money.
    token_shares: dict[str, float | None]
    cost_shares: dict[str, float | None]
    usd_per_1m_tokens: Decimal | None = None
    usd_per_1m_by_component: dict[str, Decimal | None]
    cache_hit_rate: float | None = None
    reasoning_share: float | None = None
    #: What cache reads saved against the ordinary input rate, less the premium paid
    #: to write the cache. Negative means the cache is costing more than it saves.
    cache_savings_usd: Decimal
    cache_write_premium_usd: Decimal
    net_cache_benefit_usd: Decimal
    #: Share of cached tokens whose event carried a usable rate. Below 1.0 the benefit
    #: above is a floor; ``None`` means nothing was cached.
    cache_priced_coverage: float | None = None


class UnpricedModel(BaseModel):
    model: str | None
    provider: str | None
    events: int
    tokens: int


_METRIC_FIELDS = (
    "requests",
    "subjects",
    "cost_usd",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "billable_input_tokens",
    "unpriced_events",
)

_COST_COMPONENT_FIELDS = (
    "input_usd",
    "output_usd",
    "cache_read_usd",
    "cache_write_usd",
    "reasoning_usd",
)


def _split(row: dict[str, Any], dimensions: list[str]) -> dict[str, Any]:
    """Separate dimension columns from metric columns for a stable response shape."""
    payload = {field: row.get(field) for field in _METRIC_FIELDS}
    payload["cost_components"] = {field: row.get(field) for field in _COST_COMPONENT_FIELDS}
    payload["dimensions"] = {d: row.get(d) for d in dimensions}
    return payload


@router.get("/breakdown", response_model=list[SpendRow], summary="Spend by dimension")
def breakdown(
    conn: Db,
    filters: FilterQuery,
    group_by: Annotated[
        list[str] | None, Query(description="Repeatable or comma-separated.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[dict[str, Any]]:
    dimensions = parse_group_by(group_by) or ["project"]
    rows = queries.spend_breakdown(conn, filters=filters, group_by=dimensions, limit=limit)
    return [_split(row, dimensions) for row in rows]


@router.get("/series", response_model=list[SeriesPoint], summary="Spend over time")
def series(
    conn: Db,
    filters: FilterQuery,
    granularity: Granularity = "day",
    group_by: Annotated[list[str] | None, Query()] = None,
) -> list[dict[str, Any]]:
    dimensions = parse_group_by(group_by)
    rows = queries.spend_series(conn, filters=filters, granularity=granularity, group_by=dimensions)
    return [{**_split(row, dimensions), "bucket": row["bucket"]} for row in rows]


@router.get(
    "/unit-economics",
    response_model=list[UnitEconomicsRow],
    summary="Cost per request, per customer and per 1k tokens",
)
def unit_economics(
    conn: Db,
    filters: FilterQuery,
    group_by: Annotated[list[str] | None, Query()] = None,
) -> list[dict[str, Any]]:
    dimensions = parse_group_by(group_by) or ["project"]
    rows = queries.unit_economics(conn, filters=filters, group_by=dimensions)
    return [
        {
            **_split(row, dimensions),
            "cost_per_request": row["cost_per_request"],
            "cost_per_subject": row["cost_per_subject"],
            "cost_per_1k_tokens": row["cost_per_1k_tokens"],
            "usd_per_1m_tokens": row["usd_per_1m_tokens"],
            "cache_hit_rate": row["cache_hit_rate"],
            "reasoning_share": row["reasoning_share"],
        }
        for row in rows
    ]


_TOKEN_FIELDS = (
    "total_tokens",
    "token_components",
    "token_shares",
    "cost_shares",
    "usd_per_1m_tokens",
    "usd_per_1m_by_component",
    "cache_hit_rate",
    "reasoning_share",
    "cache_savings_usd",
    "cache_write_premium_usd",
    "net_cache_benefit_usd",
    "cache_priced_coverage",
)


@router.get(
    "/tokens",
    response_model=list[TokenEconomicsRow],
    summary="Token usage and the cost of that usage",
    description=(
        "Token volume split across the five disjoint buckets the bill was actually "
        "computed from, each beside its own cost. The reported `input_tokens` and "
        "`output_tokens` are inclusive totals -- the cache and reasoning counts are "
        "already inside them -- so the components are what to compare, not the totals. "
        "Also reports what the prompt cache saved against the ordinary input rate, net "
        "of the premium paid to write it, priced from the rates each event was "
        "actually billed at rather than from today's price list."
    ),
)
def token_economics(
    conn: Db,
    filters: FilterQuery,
    group_by: Annotated[list[str] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[dict[str, Any]]:
    dimensions = parse_group_by(group_by) or ["project"]
    rows = queries.token_economics(conn, filters=filters, group_by=dimensions, limit=limit)
    return [
        {
            **_split(row, dimensions),
            **{field: derived[field] for field in _TOKEN_FIELDS},
        }
        for row, derived in ((row, token_service.derive(row)) for row in rows)
    ]


@router.get(
    "/tokens/summary",
    response_model=TokenEconomicsRow,
    summary="Window-wide token usage and cost",
    description=(
        "The same figures as `/tokens`, aggregated over the whole window in one row. "
        "This is a single ungrouped query rather than a fold of the grouped rows, so "
        "the ratios are weighted by actual volume and `subjects` stays a true distinct "
        "count instead of a sum that double-counts anyone active in two slices."
    ),
)
def token_economics_summary(conn: Db, filters: FilterQuery) -> dict[str, Any]:
    rows = queries.token_economics(conn, filters=filters, group_by=[])
    derived = token_service.derive(rows[0] if rows else {})
    return {**_split(derived, []), **{field: derived[field] for field in _TOKEN_FIELDS}}


@router.get(
    "/unpriced",
    response_model=list[UnpricedModel],
    summary="Models that could not be priced",
    description=(
        "Unpriced events are stored with a NULL cost, never zero. This endpoint is what "
        "the dashboard banner reads, so a resolution gap is visible instead of looking "
        "like a spend decrease."
    ),
)
def unpriced(conn: Db, filters: FilterQuery) -> list[dict[str, Any]]:
    return queries.unpriced_summary(conn, filters=filters)
