"""Ingestion pipeline: OTLP bytes in, priced events out."""

from __future__ import annotations

from dataclasses import dataclass, field

from tokenomics.ingest.normalize import normalize_spans
from tokenomics.ingest.otlp import decode_request, iter_spans
from tokenomics.models import UsageEvent
from tokenomics.pricing.engine import PricingEngine
from tokenomics.pricing.resolver import Method


@dataclass(slots=True)
class IngestResult:
    """What happened to one OTLP batch. Drives both the HTTP response and metrics."""

    spans_received: int = 0
    events_accepted: int = 0
    events_unpriced: int = 0
    unpriced_models: dict[str, int] = field(default_factory=dict)
    events: list[UsageEvent] = field(default_factory=list)

    @property
    def spans_ignored(self) -> int:
        """Non-GenAI spans, which are expected in a shared trace pipeline."""
        return self.spans_received - self.events_accepted


def process(body: bytes, content_type: str | None, engine: PricingEngine) -> IngestResult:
    """Decode, normalize and price one OTLP export request."""
    request = decode_request(body, content_type)
    records = list(iter_spans(request))
    result = IngestResult(spans_received=len(records))

    for event in normalize_spans(records):
        priced = engine.price(event)
        result.events.append(priced.event)
        result.events_accepted += 1

        if priced.resolution.method is Method.UNPRICED:
            result.events_unpriced += 1
            model = event.billing_model or "<unknown>"
            result.unpriced_models[model] = result.unpriced_models.get(model, 0) + 1

    return result
