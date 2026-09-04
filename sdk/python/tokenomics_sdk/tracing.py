"""Tracer setup and span emission."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

from tokenomics_sdk.attributes import (
    GEN_AI_OPERATION,
    GEN_AI_PROVIDER,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    Usage,
)
from tokenomics_sdk.context import Attribution, current, set_base

DEFAULT_ENDPOINT = "http://localhost:8000/v1/traces"
_TRACER_NAME = "tokenomics-sdk"

_provider: TracerProvider | None = None


def configure(
    *,
    endpoint: str | None = None,
    project: str | None = None,
    environment: str | None = None,
    service_name: str = "tokenomics-instrumented-app",
    batch: bool = True,
    headers: dict[str, str] | None = None,
) -> TracerProvider:
    """Point the SDK at a Tokenomics endpoint and set default attribution."""
    global _provider

    endpoint = endpoint or os.environ.get("TOKENOMICS_ENDPOINT", DEFAULT_ENDPOINT)
    set_base(Attribution(project=project, environment=environment))

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers or {})
    processor = BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    _provider = provider
    return provider


def flush(timeout_millis: int = 10_000) -> None:
    """Force-export buffered spans. Call before a short-lived process exits."""
    if _provider is not None:
        _provider.force_flush(timeout_millis)


def tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)


class SpanHandle:
    """Mutable handle to an in-flight GenAI span."""

    __slots__ = ("_span",)

    def __init__(self, span: trace.Span) -> None:
        self._span = span

    def set_usage(self, usage: Usage) -> None:
        self._span.set_attributes(dict(usage.as_attributes()))

    def set_response(self, *, model: str | None = None, response_id: str | None = None) -> None:
        if model:
            self._span.set_attribute(GEN_AI_RESPONSE_MODEL, model)
        if response_id:
            self._span.set_attribute(GEN_AI_RESPONSE_ID, response_id)

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)


@contextmanager
def llm_span(
    *,
    provider: str,
    operation: str,
    request_model: str,
    extra: dict[str, Any] | None = None,
) -> Iterator[SpanHandle]:
    """Wrap an LLM call so the span duration reflects the real call latency.

    Duration matters downstream: it feeds cost-per-second and latency-vs-cost views, and
    a zero-length span would silently break them.
    """
    attributes: dict[str, Any] = {
        GEN_AI_PROVIDER: provider,
        GEN_AI_OPERATION: operation,
        GEN_AI_REQUEST_MODEL: request_model,
    }
    attributes.update(current().as_attributes())
    if extra:
        attributes.update(extra)

    with tracer().start_as_current_span(
        f"{operation} {request_model}", kind=trace.SpanKind.CLIENT, attributes=attributes
    ) as span:
        handle = SpanHandle(span)
        try:
            yield handle
        except Exception as exc:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise


def record_call(
    *,
    provider: str,
    operation: str,
    request_model: str,
    response_model: str | None,
    usage: Usage,
    response_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a completed GenAI span in one shot (used by importers and backfills)."""
    with llm_span(
        provider=provider, operation=operation, request_model=request_model, extra=extra
    ) as span:
        span.set_response(model=response_model, response_id=response_id)
        span.set_usage(usage)
