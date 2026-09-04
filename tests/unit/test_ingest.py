"""Ingestion tests, driven by OTLP bytes the real SDK actually emitted.

Regenerate with ``uv run python tests/fixtures/generate.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenomics.ingest.normalize import normalize_span, normalize_spans
from tokenomics.ingest.otlp import (
    OtlpDecodeError,
    SpanRecord,
    decode_request,
    iter_spans,
)
from tokenomics.ingest.pipeline import process
from tokenomics.models import TokenVector

FIXTURES = Path(__file__).parent.parent / "fixtures"


def load(name: str, content_type: str):
    return decode_request((FIXTURES / name).read_bytes(), content_type)


@pytest.fixture
def events_pb():
    return list(normalize_spans(iter_spans(load("otlp_traces.pb", "application/x-protobuf"))))


def test_protobuf_and_json_encodings_agree():
    """Both OTLP wire formats must normalize identically."""
    from_pb = list(normalize_spans(iter_spans(load("otlp_traces.pb", "application/x-protobuf"))))
    from_json = list(normalize_spans(iter_spans(load("otlp_traces.json", "application/json"))))
    assert [e.model_dump() for e in from_pb] == [e.model_dump() for e in from_json]


def test_extracts_attribution(events_pb):
    by_feature = {e.attribution.feature: e for e in events_pb}
    cart = by_feature["cart-summarizer"]
    assert cart.attribution.project == "checkout"
    assert cart.attribution.environment == "prod"
    assert cart.attribution.subject_id == "cust_123"
    assert cart.attribution.prompt_version == "v4"

    support = by_feature["support-agent"]
    assert support.attribution.tags == {"tier": "enterprise"}


def test_openai_usage_is_passed_through_inclusively(events_pb):
    event = next(e for e in events_pb if e.provider == "openai")
    assert event.tokens.input == 10_000
    assert event.tokens.cache_read == 8_000
    assert event.tokens.is_consistent()


def test_anthropic_exclusive_counts_are_converted_to_inclusive(events_pb):
    """The SDK folds Anthropic's separate cache counts into the input total."""
    event = next(e for e in events_pb if e.provider == "anthropic")
    # 1200 uncached + 90000 cache read + 2000 cache write
    assert event.tokens.input == 93_200
    assert event.tokens.cache_read == 90_000
    assert event.tokens.cache_write == 2_000
    assert event.tokens.is_consistent()


def test_anthropic_round_trip_recovers_the_uncached_count(events_pb, engine):
    """Partitioning must invert the SDK's inclusive conversion exactly.

    Anthropic reported 1200 uncached input tokens. After the SDK makes the count
    inclusive and the pricing engine partitions it back apart, the billable input must
    be exactly those 1200 tokens again -- no drift, no double-count.
    """
    from tokenomics.pricing.cost import _partition_input

    event = next(e for e in events_pb if e.provider == "anthropic")
    billable, cache_read, cache_write, warnings = _partition_input(event.tokens)
    assert billable == 1_200
    assert (cache_read, cache_write) == (90_000, 2_000)
    assert warnings == []


def test_full_pipeline_prices_every_event(engine):
    result = process((FIXTURES / "otlp_traces.pb").read_bytes(), "application/x-protobuf", engine)
    assert result.spans_received == 2
    assert result.events_accepted == 2
    assert result.events_unpriced == 0
    assert all(e.cost is not None and e.cost.total_usd > 0 for e in result.events)


def test_anthropic_cost_is_hand_checkable(engine):
    """1200*3e-6 + 90000*3e-7 + 2000*3.75e-6 + 800*1.5e-5."""
    from decimal import Decimal

    result = process((FIXTURES / "otlp_traces.pb").read_bytes(), "application/x-protobuf", engine)
    event = next(e for e in result.events if e.provider == "anthropic")
    assert event.cost is not None
    assert event.cost.total_usd == Decimal("0.0501")


# --------------------------------------------------------------- alias-table coverage


def _span_with(attributes: dict) -> SpanRecord:
    from opentelemetry.proto.trace.v1.trace_pb2 import Span

    span = Span(
        trace_id=b"\x01" * 16,
        span_id=b"\x02" * 8,
        start_time_unix_nano=1_700_000_000_000_000_000,
        end_time_unix_nano=1_700_000_001_000_000_000,
    )
    record = SpanRecord(span=span, resource_attributes={}, scope_name="test")
    object.__setattr__(record, "resource_attributes", attributes)
    return record


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        (
            {
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 5,
            },
            TokenVector(input=10, output=5),
        ),
        pytest.param(
            {
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.prompt_tokens": 10,
                "gen_ai.usage.completion_tokens": 5,
            },
            TokenVector(input=10, output=5),
            id="deprecated-prompt-completion-names",
        ),
        pytest.param(
            {
                "gen_ai.request.model": "claude-sonnet-4-5",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 5,
                "gen_ai.usage.cache_creation.input_tokens": 4,
            },
            TokenVector(input=10, output=5, cache_write=4),
            id="renamed-cache_creation-still-accepted",
        ),
        pytest.param(
            {
                "gen_ai.request.model": "claude-sonnet-4-5",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 5,
                "gen_ai.usage.cache_write.input_tokens": 4,
            },
            TokenVector(input=10, output=5, cache_write=4),
            id="current-cache_write-name",
        ),
        pytest.param(
            {
                "llm.model_name": "gpt-4o",
                "llm.token_count.prompt": 10,
                "llm.token_count.completion": 5,
            },
            TokenVector(input=10, output=5),
            id="openinference-names",
        ),
    ],
)
def test_alias_table_accepts_every_instrumentation_generation(attributes, expected):
    event = normalize_span(_span_with(attributes))
    assert event is not None
    assert event.tokens == expected


def test_non_genai_spans_are_ignored():
    assert normalize_span(_span_with({"http.method": "GET", "db.system": "postgresql"})) is None


def test_genai_span_without_a_model_is_ignored():
    """Unpriceable and unattributable -- better dropped than guessed at."""
    assert normalize_span(_span_with({"gen_ai.usage.input_tokens": 10})) is None


def test_string_token_counts_are_coerced():
    event = normalize_span(
        _span_with({"gen_ai.request.model": "gpt-4o", "gen_ai.usage.input_tokens": "42"})
    )
    assert event is not None
    assert event.tokens.input == 42


# ------------------------------------------------------------------------- decoding


def test_rejects_unknown_content_type():
    with pytest.raises(OtlpDecodeError, match="unsupported content-type"):
        decode_request(b"{}", "text/csv")


def test_rejects_malformed_json():
    with pytest.raises(OtlpDecodeError, match="invalid OTLP/JSON"):
        decode_request(b"{not json", "application/json")


def test_defaults_to_protobuf_when_content_type_missing():
    request = decode_request((FIXTURES / "otlp_traces.pb").read_bytes(), None)
    assert len(list(iter_spans(request))) == 2
