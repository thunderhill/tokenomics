"""Write path: events, pricing snapshots and rollup maintenance."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from tokenomics.models import UsageEvent
from tokenomics.pricing.pricebook import PriceBook
from tokenomics.storage.database import ensure_partitions
from tokenomics.storage.queries import BILLABLE_INPUT_SQL

_INSERT = """
INSERT INTO usage_event (
    ts, trace_id, span_id, duration_ms,
    provider, request_model, response_model, model_key, operation, service_tier,
    input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens,
    cost_usd, input_usd, output_usd, cache_read_usd, cache_write_usd, reasoning_usd,
    pricing_snapshot_id, rates_applied, context_tier, cost_warnings,
    project, feature, environment, subject_id, prompt_version, tags
) VALUES (
    %(ts)s, %(trace_id)s, %(span_id)s, %(duration_ms)s,
    %(provider)s, %(request_model)s, %(response_model)s, %(model_key)s, %(operation)s,
    %(service_tier)s,
    %(input_tokens)s, %(output_tokens)s, %(cache_read_tokens)s, %(cache_write_tokens)s,
    %(reasoning_tokens)s,
    %(cost_usd)s, %(input_usd)s, %(output_usd)s, %(cache_read_usd)s, %(cache_write_usd)s,
    %(reasoning_usd)s,
    %(pricing_snapshot_id)s, %(rates_applied)s, %(context_tier)s, %(cost_warnings)s,
    %(project)s, %(feature)s, %(environment)s, %(subject_id)s, %(prompt_version)s, %(tags)s
)
ON CONFLICT (ts, trace_id, span_id) DO NOTHING
"""


def _row(event: UsageEvent) -> dict[str, Any]:
    cost = event.cost
    tokens = event.tokens
    return {
        "ts": event.ts,
        "trace_id": event.trace_id,
        "span_id": event.span_id,
        "duration_ms": event.duration_ms,
        "provider": event.provider,
        "request_model": event.request_model,
        "response_model": event.response_model,
        "model_key": cost.model_key if cost else None,
        "operation": event.operation,
        "service_tier": str(event.service_tier),
        "input_tokens": tokens.input,
        "output_tokens": tokens.output,
        "cache_read_tokens": tokens.cache_read,
        "cache_write_tokens": tokens.cache_write,
        "reasoning_tokens": tokens.reasoning,
        # All cost columns stay NULL for unpriced events -- never 0.
        "cost_usd": cost.total_usd if cost else None,
        "input_usd": cost.input_usd if cost else None,
        "output_usd": cost.output_usd if cost else None,
        "cache_read_usd": cost.cache_read_usd if cost else None,
        "cache_write_usd": cost.cache_write_usd if cost else None,
        "reasoning_usd": cost.reasoning_usd if cost else None,
        "pricing_snapshot_id": cost.snapshot_id if cost else None,
        "rates_applied": Jsonb({k: str(v) for k, v in cost.rates_applied.items()})
        if cost
        else None,
        "context_tier": cost.context_tier if cost else None,
        "cost_warnings": list(cost.warnings) if cost else [],
        "project": event.attribution.project,
        "feature": event.attribution.feature,
        "environment": event.attribution.environment,
        "subject_id": event.attribution.subject_id,
        "prompt_version": event.attribution.prompt_version,
        "tags": Jsonb(event.attribution.tags),
    }


def insert_events(conn: psycopg.Connection, events: list[UsageEvent]) -> int:
    """Insert events idempotently. Returns the number actually written.

    Duplicate (ts, trace_id, span_id) rows are dropped, so collector retries and
    re-running an importer cannot double-bill.
    """
    if not events:
        return 0

    ensure_partitions(conn, [event.ts for event in events])
    rows = [_row(event) for event in events]

    with conn.cursor() as cur:
        cur.executemany(_INSERT, rows)
        written = cur.rowcount
    return max(written, 0)


def record_snapshot(conn: psycopg.Connection, book: PriceBook) -> None:
    """Register a pricing snapshot so historical costs stay auditable."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pricing_snapshot (snapshot_id, source, fetched_at, model_count) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (snapshot_id) DO NOTHING",
            (book.snapshot_id, book.source, book.fetched_at, len(book)),
        )


# The billable-input expression is shared with the read path rather than restated, so
# the rollup cannot come to disagree with the query it is meant to accelerate.
_REFRESH_ROLLUP = f"""
INSERT INTO usage_rollup_hourly (
    bucket, project, feature, environment, model_key, provider,
    requests, subjects, input_tokens, output_tokens,
    cache_read_tokens, cache_write_tokens, reasoning_tokens, billable_input_tokens,
    cost_usd, input_usd, output_usd, cache_read_usd, cache_write_usd, reasoning_usd,
    unpriced_events, refreshed_at
)
SELECT
    date_trunc('hour', ts)          AS bucket,
    project,
    coalesce(feature, '')           AS feature,
    coalesce(environment, '')       AS environment,
    coalesce(model_key, '')         AS model_key,
    coalesce(provider, '')          AS provider,
    count(*)                        AS requests,
    count(DISTINCT subject_id)      AS subjects,
    sum(input_tokens),
    sum(output_tokens),
    sum(cache_read_tokens),
    sum(cache_write_tokens),
    sum(reasoning_tokens),
    sum({BILLABLE_INPUT_SQL}),
    coalesce(sum(cost_usd), 0)      AS cost_usd,
    coalesce(sum(input_usd), 0),
    coalesce(sum(output_usd), 0),
    coalesce(sum(cache_read_usd), 0),
    coalesce(sum(cache_write_usd), 0),
    coalesce(sum(reasoning_usd), 0),
    count(*) FILTER (WHERE cost_usd IS NULL) AS unpriced_events,
    now()
FROM usage_event
WHERE ts >= %(since)s AND ts < %(until)s
GROUP BY 1, 2, 3, 4, 5, 6
ON CONFLICT (bucket, project, feature, environment, model_key, provider)
DO UPDATE SET
    requests              = EXCLUDED.requests,
    subjects              = EXCLUDED.subjects,
    input_tokens          = EXCLUDED.input_tokens,
    output_tokens         = EXCLUDED.output_tokens,
    cache_read_tokens     = EXCLUDED.cache_read_tokens,
    cache_write_tokens    = EXCLUDED.cache_write_tokens,
    reasoning_tokens      = EXCLUDED.reasoning_tokens,
    billable_input_tokens = EXCLUDED.billable_input_tokens,
    cost_usd              = EXCLUDED.cost_usd,
    input_usd             = EXCLUDED.input_usd,
    output_usd            = EXCLUDED.output_usd,
    cache_read_usd        = EXCLUDED.cache_read_usd,
    cache_write_usd       = EXCLUDED.cache_write_usd,
    reasoning_usd         = EXCLUDED.reasoning_usd,
    unpriced_events       = EXCLUDED.unpriced_events,
    refreshed_at          = now()
"""


def refresh_rollups(conn: psycopg.Connection, since: datetime, until: datetime) -> int:
    """Recompute hourly rollups for a window.

    Recompute-and-upsert rather than incremental addition: late-arriving spans are
    normal in tracing, and an additive rollup would drift permanently out of step with
    the events table.
    """
    with conn.cursor() as cur:
        cur.execute(_REFRESH_ROLLUP, {"since": since, "until": until})
        return max(cur.rowcount, 0)


def total_spend(conn: psycopg.Connection, since: datetime, until: datetime) -> Decimal:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT coalesce(sum(cost_usd), 0) FROM usage_event WHERE ts >= %s AND ts < %s",
            (since, until),
        )
        row = cur.fetchone()
    return Decimal(row[0]) if row and row[0] is not None else Decimal(0)
