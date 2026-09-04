"""Read path: cost aggregation by any combination of attribution tags.

Group-by dimensions are composed as SQL identifiers, so they come from a fixed
whitelist -- never from raw request strings. Values are always bound parameters.

Unpriced events are counted separately in every result rather than folded into the cost
sum. ``sum(cost_usd)`` skips NULLs, so without that count a resolution gap would look
like a spend *decrease*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

#: Columns that may appear in GROUP BY. Anything else is rejected.
DIMENSIONS: frozenset[str] = frozenset(
    {
        "project",
        "feature",
        "environment",
        "subject_id",
        "prompt_version",
        "model_key",
        "provider",
        "operation",
        "service_tier",
    }
)

Granularity = Literal["hour", "day", "week", "month"]
_GRANULARITIES: frozenset[str] = frozenset({"hour", "day", "week", "month"})


class InvalidDimensionError(ValueError):
    """A requested group-by dimension is not in the whitelist."""


@dataclass(frozen=True, slots=True)
class Filters:
    """Query filters. Every field becomes a bound parameter."""

    since: datetime
    until: datetime
    project: tuple[str, ...] = ()
    feature: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    subject_id: tuple[str, ...] = ()
    prompt_version: tuple[str, ...] = ()
    model_key: tuple[str, ...] = ()
    provider: tuple[str, ...] = ()
    tags: dict[str, str] = field(default_factory=dict)

    def where(self) -> tuple[sql.Composed, dict[str, Any]]:
        clauses: list[sql.Composable] = [sql.SQL("ts >= %(since)s AND ts < %(until)s")]
        params: dict[str, Any] = {"since": self.since, "until": self.until}

        for name in (
            "project",
            "feature",
            "environment",
            "subject_id",
            "prompt_version",
            "model_key",
            "provider",
        ):
            values = getattr(self, name)
            if values:
                clauses.append(
                    sql.SQL("{} = ANY(%({})s)").format(sql.Identifier(name), sql.SQL(name))
                )
                params[name] = list(values)

        if self.tags:
            clauses.append(sql.SQL("tags @> %(tags)s"))
            params["tags"] = Jsonb(self.tags)

        return sql.SQL(" AND ").join(clauses), params


def _validate(dimensions: list[str]) -> list[str]:
    for dimension in dimensions:
        if dimension not in DIMENSIONS:
            raise InvalidDimensionError(
                f"unknown dimension {dimension!r}; allowed: {sorted(DIMENSIONS)}"
            )
    return dimensions


# --------------------------------------------------------------------------- partition
#
# The aggregate form of the partition in ``tokenomics.pricing.cost._partition_input``,
# which is the authority: these expressions must produce exactly what that function
# produced at ingest, or the token split and the cost split describe different events.
# ``tests/integration/test_storage.py`` cross-checks the two, so the duplication cannot
# drift silently.
#
# The subtle case is inconsistent instrumentation -- a provider reporting more cached
# tokens than input tokens. Clamping the billable remainder to zero and leaving the
# cache counts raw would leave the three buckets summing to *more* than the input that
# was actually billed, so a token bar built from them would not add up. Scaling the
# components down proportionally, the way the cost function does, keeps it exact.

# Plain strings, because the rollup refresh in ``storage.repository`` needs the same
# expressions in a query it composes itself. They contain no user input -- only column
# names this module owns -- so interpolating them is safe.
_CACHED_SQL = "(cache_read_tokens + cache_write_tokens)"

_EFF_CACHE_READ_SQL = (
    f"CASE WHEN {_CACHED_SQL} > input_tokens "
    f"THEN trunc(cache_read_tokens::numeric * input_tokens / {_CACHED_SQL})::bigint "
    "ELSE cache_read_tokens END"
)

_EFF_CACHE_WRITE_SQL = (
    f"CASE WHEN {_CACHED_SQL} > input_tokens "
    f"THEN least(input_tokens - ({_EFF_CACHE_READ_SQL}), "
    f"trunc(cache_write_tokens::numeric * input_tokens / {_CACHED_SQL})::bigint) "
    "ELSE cache_write_tokens END"
)

#: Input billed at the full input rate: the total less whatever came from cache.
BILLABLE_INPUT_SQL = (
    f"greatest(input_tokens - ({_EFF_CACHE_READ_SQL}) - ({_EFF_CACHE_WRITE_SQL}), 0)"
)

_EFF_CACHE_READ = sql.SQL(_EFF_CACHE_READ_SQL)
_EFF_CACHE_WRITE = sql.SQL(_EFF_CACHE_WRITE_SQL)
_BILLABLE_INPUT = sql.SQL(BILLABLE_INPUT_SQL)

#: Reasoning is a subset of output, and the cost function clamps it before billing.
_EFF_REASONING = sql.SQL("least(reasoning_tokens, output_tokens)")

_METRICS = sql.SQL(
    """
    count(*)                                   AS requests,
    count(DISTINCT subject_id)                 AS subjects,
    coalesce(sum(cost_usd), 0)                 AS cost_usd,
    coalesce(sum(input_tokens), 0)             AS input_tokens,
    coalesce(sum(output_tokens), 0)            AS output_tokens,
    coalesce(sum(cache_read_tokens), 0)        AS cache_read_tokens,
    coalesce(sum(cache_write_tokens), 0)       AS cache_write_tokens,
    coalesce(sum(reasoning_tokens), 0)         AS reasoning_tokens,
    coalesce(sum({billable_input}), 0)         AS billable_input_tokens,
    coalesce(sum(input_usd), 0)                AS input_usd,
    coalesce(sum(output_usd), 0)               AS output_usd,
    coalesce(sum(cache_read_usd), 0)           AS cache_read_usd,
    coalesce(sum(cache_write_usd), 0)          AS cache_write_usd,
    coalesce(sum(reasoning_usd), 0)            AS reasoning_usd,
    count(*) FILTER (WHERE cost_usd IS NULL)   AS unpriced_events
    """
).format(billable_input=_BILLABLE_INPUT)


def spend_breakdown(
    conn: psycopg.Connection,
    *,
    filters: Filters,
    group_by: list[str],
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Aggregate spend grouped by any combination of whitelisted dimensions."""
    dimensions = _validate(group_by)
    where, params = filters.where()

    if dimensions:
        columns = sql.SQL(", ").join(sql.Identifier(d) for d in dimensions)
        select = sql.SQL(
            "SELECT {cols}, {metrics} FROM usage_event WHERE {where} "
            "GROUP BY {cols} ORDER BY cost_usd DESC LIMIT %(limit)s"
        )
        query = select.format(cols=columns, metrics=_METRICS, where=where)
    else:
        query = sql.SQL("SELECT {metrics} FROM usage_event WHERE {where} LIMIT %(limit)s").format(
            metrics=_METRICS, where=where
        )

    params["limit"] = limit
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return list(cur.fetchall())


def spend_series(
    conn: psycopg.Connection,
    *,
    filters: Filters,
    granularity: Granularity = "day",
    group_by: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Spend over time, optionally split by dimensions."""
    if granularity not in _GRANULARITIES:
        raise InvalidDimensionError(f"unknown granularity {granularity!r}")
    dimensions = _validate(group_by or [])
    where, params = filters.where()

    bucket = sql.SQL("date_trunc({}, ts)").format(sql.Literal(granularity))
    group_columns: list[sql.Composable] = [bucket]
    select_columns: list[sql.Composable] = [sql.SQL("{} AS bucket").format(bucket)]
    for dimension in dimensions:
        group_columns.append(sql.Identifier(dimension))
        select_columns.append(sql.Identifier(dimension))

    query = sql.SQL(
        "SELECT {select}, {metrics} FROM usage_event WHERE {where} GROUP BY {group} ORDER BY bucket"
    ).format(
        select=sql.SQL(", ").join(select_columns),
        metrics=_METRICS,
        where=where,
        group=sql.SQL(", ").join(group_columns),
    )

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return list(cur.fetchall())


def unit_economics(
    conn: psycopg.Connection, *, filters: Filters, group_by: list[str] | None = None
) -> list[dict[str, Any]]:
    """Cost per request and cost per distinct subject."""
    rows = spend_breakdown(conn, filters=filters, group_by=group_by or ["project"], limit=500)
    for row in rows:
        cost = Decimal(row["cost_usd"])
        requests = row["requests"] or 0
        subjects = row["subjects"] or 0
        row["cost_per_request"] = (cost / requests) if requests else Decimal(0)
        row["cost_per_subject"] = (cost / subjects) if subjects else None
        tokens = (row["input_tokens"] or 0) + (row["output_tokens"] or 0)
        row["cost_per_1k_tokens"] = (cost / Decimal(tokens) * 1000) if tokens else Decimal(0)
        # $/1M is how every provider quotes a rate, so it is the figure you can compare
        # against a price list without arithmetic. A slice with nothing priced in it
        # reports None: "$0.00 per million" would claim the tokens were free.
        priced = row["requests"] and row["unpriced_events"] < row["requests"]
        row["usd_per_1m_tokens"] = (
            (cost / Decimal(tokens) * 1_000_000) if (tokens and priced) else None
        )
        row["cache_hit_rate"] = (
            row["cache_read_tokens"] / row["input_tokens"] if row["input_tokens"] else 0.0
        )
        output = row["output_tokens"] or 0
        row["reasoning_share"] = (row["reasoning_tokens"] / output) if output else None
    return rows


#: Reasoning tokens that were actually billed apart from output.
#:
#: ``reasoning_tokens`` is a *subset* of ``output_tokens``, and only the ~58 models
#: with their own reasoning rate bill it separately -- for everyone else it is already
#: inside the output charge. Splitting the token bar on the reported count would
#: therefore show volume leaving the output slice while its cost stayed behind,
#: inflating the apparent output rate. ``reasoning_usd > 0`` is the per-row record of
#: which way the event was actually billed, so the token split follows the money.
_BILLED_REASONING = sql.SQL("CASE WHEN reasoning_usd > 0 THEN {reasoning} ELSE 0 END").format(
    reasoning=_EFF_REASONING
)


#: What the cached tokens *would* have cost at the ordinary input rate.
#:
#: Every priced event stores the exact rates it was billed at in ``rates_applied``, so
#: the counterfactual is read back per event rather than re-derived from today's price
#: list -- a cache saving computed against a rate that has since changed would be
#: fiction. Events we could not price carry no rates, so they are excluded from the
#: sums *and* counted in ``cache_basis_tokens``: a partially-covered answer says so
#: rather than quietly understating the saving.
_CACHE_COUNTERFACTUAL = sql.SQL(
    """
    coalesce(sum(cache_read_tokens * (rates_applied->>'input')::numeric)
             FILTER (WHERE rates_applied->>'input' IS NOT NULL), 0)
                                               AS cache_read_at_input_usd,
    coalesce(sum(cache_write_tokens * (rates_applied->>'input')::numeric)
             FILTER (WHERE rates_applied->>'input' IS NOT NULL), 0)
                                               AS cache_write_at_input_usd,
    coalesce(sum(cache_read_tokens + cache_write_tokens)
             FILTER (WHERE rates_applied->>'input' IS NOT NULL), 0)
                                               AS cache_basis_tokens,
    coalesce(sum(cache_read_tokens + cache_write_tokens), 0)
                                               AS cache_tokens,
    coalesce(sum({eff_read}), 0)               AS billable_cache_read_tokens,
    coalesce(sum({eff_write}), 0)              AS billable_cache_write_tokens,
    coalesce(sum(output_tokens - {billed_reasoning}), 0)
                                               AS billable_output_tokens,
    coalesce(sum({billed_reasoning}), 0)       AS billed_reasoning_tokens
    """
).format(
    billed_reasoning=_BILLED_REASONING,
    eff_read=_EFF_CACHE_READ,
    eff_write=_EFF_CACHE_WRITE,
)


def token_economics(
    conn: psycopg.Connection,
    *,
    filters: Filters,
    group_by: list[str] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Token volume and the cost of that volume, split by disjoint billable component.

    Returns the raw sums only. The derived figures -- shares, blended rates, what the
    cache actually saved -- are computed by :mod:`tokenomics.finops.tokens`, which is
    pure and therefore testable without a database.
    """
    dimensions = _validate(group_by or [])
    where, params = filters.where()
    params["limit"] = limit

    metrics = sql.SQL("{}, {}").format(_METRICS, _CACHE_COUNTERFACTUAL)

    if dimensions:
        columns = sql.SQL(", ").join(sql.Identifier(d) for d in dimensions)
        query = sql.SQL(
            "SELECT {cols}, {metrics} FROM usage_event WHERE {where} "
            "GROUP BY {cols} ORDER BY cost_usd DESC LIMIT %(limit)s"
        ).format(cols=columns, metrics=metrics, where=where)
    else:
        query = sql.SQL("SELECT {metrics} FROM usage_event WHERE {where} LIMIT %(limit)s").format(
            metrics=metrics, where=where
        )

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return list(cur.fetchall())


def unpriced_summary(conn: psycopg.Connection, *, filters: Filters) -> list[dict[str, Any]]:
    """Models we failed to price, so the gap is visible instead of silently zero."""
    where, params = filters.where()
    query = sql.SQL(
        "SELECT coalesce(response_model, request_model) AS model, provider, count(*) AS events, "
        "sum(input_tokens + output_tokens) AS tokens "
        "FROM usage_event WHERE {where} AND cost_usd IS NULL "
        "GROUP BY 1, 2 ORDER BY events DESC LIMIT 50"
    ).format(where=where)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return list(cur.fetchall())


def daily_spend(conn: psycopg.Connection, *, filters: Filters) -> list[tuple[datetime, Decimal]]:
    """Compact daily series used by forecasting."""
    rows = spend_series(conn, filters=filters, granularity="day")
    return [(row["bucket"], Decimal(row["cost_usd"])) for row in rows]


def hourly_spend(conn: psycopg.Connection, *, filters: Filters) -> list[tuple[datetime, Decimal]]:
    """Compact hourly series used by anomaly detection."""
    rows = spend_series(conn, filters=filters, granularity="hour")
    return [(row["bucket"], Decimal(row["cost_usd"])) for row in rows]
