"""Regenerate recorded OTLP fixtures.

Fixtures are produced by driving the real SDK against fake provider clients and
capturing the actual OTLP protobuf the exporter would send. That keeps the ingestion
tests honest: they parse bytes the SDK genuinely emits, not bytes we hand-wrote to
match our own parser.

    uv run python tests/fixtures/generate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

FIXTURES = Path(__file__).parent
sys.path.insert(0, str(FIXTURES.parent.parent / "sdk" / "python"))

from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans  # noqa: E402
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from tokenomics_sdk import context as ctx  # noqa: E402
from tokenomics_sdk import tracing, track, wrap_anthropic, wrap_openai  # noqa: E402


class FakeUsage(dict):
    __getattr__ = dict.get  # type: ignore[assignment]


class FakeResponse:
    def __init__(self, model: str, usage: dict, response_id: str) -> None:
        self.model = model
        self.usage = FakeUsage(usage)
        self.id = response_id


class _OpenAICompletions:
    @staticmethod
    def create(**kwargs: object) -> FakeResponse:
        return FakeResponse(
            "gpt-4o-2024-08-06",
            {
                "prompt_tokens": 10_000,
                "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 8_000},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            "chatcmpl-fixture-1",
        )


class _OpenAIChat:
    completions = _OpenAICompletions()


class FakeOpenAI:
    """Mimics the shape `wrap_openai` patches, with OpenAI's *inclusive* usage counts."""

    chat = _OpenAIChat()


class _AnthropicMessages:
    @staticmethod
    def create(**kwargs: object) -> FakeResponse:
        return FakeResponse(
            "claude-sonnet-4-5-20250929",
            {
                "input_tokens": 1_200,
                "output_tokens": 800,
                "cache_read_input_tokens": 90_000,
                "cache_creation_input_tokens": 2_000,
            },
            "msg_fixture_1",
        )


class FakeAnthropic:
    """Mimics the Anthropic client, with its *exclusive* usage counts."""

    messages = _AnthropicMessages()


def main() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "fixture-app"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    from opentelemetry import trace

    trace.set_tracer_provider(provider)
    tracing._provider = provider
    ctx.set_base(ctx.Attribution(project="checkout", environment="prod"))

    openai_client = wrap_openai(FakeOpenAI())
    anthropic_client = wrap_anthropic(FakeAnthropic())

    with track(feature="cart-summarizer", subject_id="cust_123", prompt_version="v4"):
        openai_client.chat.completions.create(model="gpt-4o", messages=[])

    with track(feature="support-agent", subject_id="cust_456", tier="enterprise"):
        anthropic_client.messages.create(model="claude-sonnet-4-5", messages=[])

    spans = exporter.get_finished_spans()
    assert len(spans) == 2, f"expected 2 spans, got {len(spans)}"

    request = encode_spans(spans)
    (FIXTURES / "otlp_traces.pb").write_bytes(request.SerializeToString())

    from google.protobuf import json_format

    (FIXTURES / "otlp_traces.json").write_text(json_format.MessageToJson(request, indent=2) + "\n")
    print(f"wrote {len(spans)} spans to otlp_traces.pb and otlp_traces.json")


if __name__ == "__main__":
    main()
