"""Normalize OTel GenAI spans into :class:`UsageEvent`.

Everything provider-, vendor- and spec-version-specific is resolved here, so that
nothing downstream has to know which instrumentation produced a span.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import Any

from tokenomics.ingest.aliases import (
    ATTRIBUTION_ALIASES,
    IDENTITY_ALIASES,
    TAG_PREFIX,
    USAGE_ALIASES,
    coerce_int,
    first_present,
)
from tokenomics.ingest.otlp import SpanRecord
from tokenomics.models import Attribution, ServiceTier, TokenVector, UsageEvent

_SERVICE_TIERS: dict[str, ServiceTier] = {
    "default": ServiceTier.STANDARD,
    "standard": ServiceTier.STANDARD,
    "auto": ServiceTier.STANDARD,
    "scale": ServiceTier.STANDARD,
    "priority": ServiceTier.PRIORITY,
    "flex": ServiceTier.FLEX,
    "batch": ServiceTier.BATCH,
    "batches": ServiceTier.BATCH,
}

#: Span attribute keys that indicate a GenAI call worth costing.
_GENAI_MARKERS = ("gen_ai.", "llm.")


def _is_genai_span(attributes: dict[str, Any]) -> bool:
    return any(key.startswith(_GENAI_MARKERS) for key in attributes)


def _service_tier(raw: Any) -> ServiceTier:
    if not isinstance(raw, str):
        return ServiceTier.STANDARD
    return _SERVICE_TIERS.get(raw.strip().lower(), ServiceTier.STANDARD)


def _token_vector(attributes: dict[str, Any]) -> TokenVector:
    counts = {
        field: coerce_int(first_present(attributes, aliases))
        for field, aliases in USAGE_ALIASES.items()
    }
    return TokenVector(**counts)


def _attribution(attributes: dict[str, Any]) -> Attribution:
    values: dict[str, Any] = {}
    for field, aliases in ATTRIBUTION_ALIASES.items():
        found = first_present(attributes, aliases)
        if found is not None:
            values[field] = str(found)

    tags = {
        key.removeprefix(TAG_PREFIX): str(value)
        for key, value in attributes.items()
        if key.startswith(TAG_PREFIX)
    }
    return Attribution(**values, tags=tags)


def _timestamp(nanos: int) -> datetime:
    return datetime.fromtimestamp(nanos / 1_000_000_000, tz=UTC)


def normalize_span(record: SpanRecord) -> UsageEvent | None:
    """Convert one span to a :class:`UsageEvent`, or ``None`` if it is not a GenAI call."""
    attributes = record.attributes
    if not _is_genai_span(attributes):
        return None

    request_model = first_present(attributes, IDENTITY_ALIASES["request_model"])
    response_model = first_present(attributes, IDENTITY_ALIASES["response_model"])
    if not (request_model or response_model):
        # A GenAI span with no model cannot be priced; ignore it rather than guess.
        return None

    span = record.span
    duration_ns = span.end_time_unix_nano - span.start_time_unix_nano

    return UsageEvent(
        trace_id=span.trace_id.hex(),
        span_id=span.span_id.hex(),
        ts=_timestamp(span.start_time_unix_nano),
        duration_ms=(duration_ns / 1_000_000) if duration_ns > 0 else None,
        provider=_optional_str(first_present(attributes, IDENTITY_ALIASES["provider"])),
        request_model=_optional_str(request_model),
        response_model=_optional_str(response_model),
        operation=_optional_str(first_present(attributes, IDENTITY_ALIASES["operation"])) or "chat",
        service_tier=_service_tier(first_present(attributes, IDENTITY_ALIASES["service_tier"])),
        tokens=_token_vector(attributes),
        attribution=_attribution(attributes),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_spans(records: Iterable[SpanRecord]) -> Iterator[UsageEvent]:
    for record in records:
        if (event := normalize_span(record)) is not None:
            yield event
