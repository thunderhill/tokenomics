"""Prometheus metrics.

Label cardinality is the thing that kills a metrics endpoint, and LLM telemetry is full
of unbounded strings -- subject ids, prompt versions, trace ids. So the counters here
carry only bounded labels (provider, project, component) and deliberately **not** model
name on the money counters: a price list with 3000 keys times a few projects is a bad
trade for something the API can already group by.

The exception is ``tokenomics_unpriced_events_total``, which is labelled by model on
purpose. That metric exists to be alerted on, and the model name *is* the actionable
part -- "we are flying blind on claude-opus-4-6" is the whole message.
"""

from __future__ import annotations

from datetime import UTC, datetime

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from tokenomics.ingest.pipeline import IngestResult
from tokenomics.models import UsageEvent

REGISTRY = CollectorRegistry(auto_describe=True)

spans_received = Counter("tokenomics_spans_received", "OTLP spans received.", registry=REGISTRY)
spans_ignored = Counter(
    "tokenomics_spans_ignored",
    "Spans that carried no GenAI usage and were skipped.",
    registry=REGISTRY,
)
events_ingested = Counter(
    "tokenomics_events_ingested",
    "Usage events accepted.",
    ["provider", "project"],
    registry=REGISTRY,
)
unpriced_events = Counter(
    "tokenomics_unpriced_events",
    "Events we could not price. Labelled by model because the model is the fix.",
    ["model"],
    registry=REGISTRY,
)
tokens = Counter(
    "tokenomics_tokens",
    "Tokens by component. Components are disjoint: input excludes cache read/write.",
    ["component", "provider"],
    registry=REGISTRY,
)
cost_usd = Counter(
    "tokenomics_cost_usd",
    "Attributed spend in USD.",
    ["provider", "project"],
    registry=REGISTRY,
)
cost_usd_by_component = Counter(
    "tokenomics_cost_usd_by_component",
    "Spend in USD by token component -- the money twin of tokenomics_tokens, "
    "over the same disjoint partition, so volume and cost can be divided.",
    ["component", "provider"],
    registry=REGISTRY,
)
ingest_lag = Histogram(
    "tokenomics_ingest_lag_seconds",
    "Delay between a call happening and its span being ingested.",
    buckets=(1, 5, 15, 60, 300, 900, 3600, 21600, 86400),
    registry=REGISTRY,
)
ingest_duration = Histogram(
    "tokenomics_ingest_duration_seconds",
    "Time spent decoding, pricing and storing one OTLP batch.",
    registry=REGISTRY,
)
budget_utilization = Gauge(
    "tokenomics_budget_utilization",
    "Spend divided by limit for the current period. Above 1 means over budget.",
    ["budget"],
    registry=REGISTRY,
)
anomalies_detected = Counter(
    "tokenomics_anomalies_detected", "Spend anomalies flagged.", registry=REGISTRY
)
pricebook_models = Gauge(
    "tokenomics_pricebook_models",
    "Models in the active pricing snapshot.",
    ["snapshot"],
    registry=REGISTRY,
)


def observe_ingest(result: IngestResult, *, duration_seconds: float | None = None) -> None:
    """Record one ingest batch."""
    spans_received.inc(result.spans_received)
    spans_ignored.inc(result.spans_ignored)
    if duration_seconds is not None:
        ingest_duration.observe(duration_seconds)

    now = datetime.now(UTC)
    for event in result.events:
        _observe_event(event, now)

    for model, count in result.unpriced_models.items():
        unpriced_events.labels(model=model).inc(count)


def _observe_event(event: UsageEvent, now: datetime) -> None:
    provider = event.provider or "unknown"
    project = event.attribution.project

    events_ingested.labels(provider=provider, project=project).inc()

    vector = event.tokens
    # Report the same disjoint partition the cost function bills, so the metrics and
    # the invoice tell the same story.
    billable_input = max(vector.input - vector.cache_read - vector.cache_write, 0)
    for component, value in (
        ("input", billable_input),
        ("output", vector.output),
        ("cache_read", vector.cache_read),
        ("cache_write", vector.cache_write),
        ("reasoning", vector.reasoning),
    ):
        if value:
            tokens.labels(component=component, provider=provider).inc(value)

    if event.cost is not None:
        cost_usd.labels(provider=provider, project=project).inc(float(event.cost.total_usd))
        # Same components, same order, same partition as the token loop above, so
        # dividing one series by the other gives a real effective rate.
        for component, amount in event.cost.billable_units.items():
            if amount:
                cost_usd_by_component.labels(component=component, provider=provider).inc(
                    float(amount)
                )

    lag = (now - event.ts).total_seconds()
    if lag >= 0:
        ingest_lag.observe(lag)


def render() -> tuple[bytes, str]:
    """The /metrics payload and its content type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
