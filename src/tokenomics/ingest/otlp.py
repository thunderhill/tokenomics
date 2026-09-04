"""OTLP/HTTP decoding.

Accepts both protobuf and JSON encodings. JSON payloads are parsed *into* the protobuf
message so downstream code has a single representation to reason about, rather than two
parallel field-name conventions (OTLP JSON uses camelCase and boxed ``AnyValue``s).

Native OTLP **gRPC** is intentionally not implemented here. The supported path is an
OpenTelemetry Collector (``deploy/otel-collector-config.yaml``) receiving gRPC and
exporting OTLP/HTTP to this endpoint, which also brings batching, retry and backpressure
for free. Reimplementing that would be reinventing a well-maintained wheel.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from google.protobuf import json_format
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import Span

PROTOBUF_CONTENT_TYPES = frozenset(
    {"application/x-protobuf", "application/protobuf", "application/octet-stream"}
)
JSON_CONTENT_TYPES = frozenset({"application/json"})


class OtlpDecodeError(ValueError):
    """The request body was not a decodable OTLP trace export."""


def decode_request(body: bytes, content_type: str | None) -> ExportTraceServiceRequest:
    """Decode an OTLP/HTTP trace export body."""
    media_type = (content_type or "application/x-protobuf").split(";")[0].strip().lower()
    request = ExportTraceServiceRequest()

    if media_type in JSON_CONTENT_TYPES:
        try:
            json_format.Parse(body.decode("utf-8"), request)
        except (json_format.ParseError, UnicodeDecodeError) as exc:
            raise OtlpDecodeError(f"invalid OTLP/JSON: {exc}") from exc
        return request

    if media_type in PROTOBUF_CONTENT_TYPES:
        try:
            request.ParseFromString(body)
        except DecodeError as exc:
            raise OtlpDecodeError(f"invalid OTLP/protobuf: {exc}") from exc
        return request

    raise OtlpDecodeError(f"unsupported content-type: {media_type!r}")


def any_value(value: AnyValue) -> Any:
    """Unbox an OTLP ``AnyValue`` into a plain Python value."""
    which = value.WhichOneof("value")
    if which is None:
        return None
    if which == "array_value":
        return [any_value(item) for item in value.array_value.values]
    if which == "kvlist_value":
        return {kv.key: any_value(kv.value) for kv in value.kvlist_value.values}
    if which == "bytes_value":
        return value.bytes_value.hex()
    return getattr(value, which)


def attributes_to_dict(attributes: list[KeyValue]) -> dict[str, Any]:
    return {kv.key: any_value(kv.value) for kv in attributes}


@dataclass(frozen=True, slots=True)
class SpanRecord:
    """One span plus the resource attributes it inherits."""

    span: Span
    resource_attributes: dict[str, Any]
    scope_name: str

    @property
    def attributes(self) -> dict[str, Any]:
        """Span attributes layered over resource attributes (span wins)."""
        merged = dict(self.resource_attributes)
        merged.update(attributes_to_dict(list(self.span.attributes)))
        return merged


def iter_spans(request: ExportTraceServiceRequest) -> Iterator[SpanRecord]:
    """Flatten the resource/scope/span nesting into individual records."""
    for resource_spans in request.resource_spans:
        resource_attributes = attributes_to_dict(list(resource_spans.resource.attributes))
        for scope_spans in resource_spans.scope_spans:
            scope_name = scope_spans.scope.name if scope_spans.HasField("scope") else ""
            for span in scope_spans.spans:
                yield SpanRecord(
                    span=span,
                    resource_attributes=resource_attributes,
                    scope_name=scope_name,
                )
