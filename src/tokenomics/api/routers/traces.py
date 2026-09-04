"""OTLP/HTTP trace ingestion.

Accepts the same payloads any OTel exporter already sends, in protobuf or JSON, and
answers in the encoding the client used. gRPC is deliberately not implemented here --
``deploy/otel-collector-config.yaml`` runs a Collector that converts gRPC to OTLP/HTTP,
which is the supported way to get gRPC, batching and retries without reimplementing them.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response, status
from google.protobuf import json_format
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTracePartialSuccess,
    ExportTraceServiceResponse,
)

from tokenomics.api.deps import Config, Db, Engine
from tokenomics.ingest.otlp import PROTOBUF_CONTENT_TYPES, OtlpDecodeError
from tokenomics.ingest.pipeline import process
from tokenomics.storage import database, repository
from tokenomics.telemetry import metrics

router = APIRouter(tags=["ingest"])


@router.post(
    "/v1/traces",
    summary="OTLP/HTTP trace ingestion",
    response_description="OTLP ExportTraceServiceResponse, matching the request encoding.",
)
async def export_traces(request: Request, conn: Db, engine: Engine, settings: Config) -> Response:
    body = await request.body()
    if len(body) > settings.max_body_bytes:
        return _error(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"body exceeds {settings.max_body_bytes} bytes",
        )

    content_type = request.headers.get("content-type")
    started = time.perf_counter()
    try:
        result = process(body, content_type, engine)
    except OtlpDecodeError as exc:
        # 400: the exporter should not retry a payload we cannot parse.
        return _error(status.HTTP_400_BAD_REQUEST, str(exc))

    if result.events:
        # Events can arrive for any month (backfills, long queues), so make sure the
        # partition exists before writing rather than failing the batch.
        database.ensure_partitions(conn, [event.ts for event in result.events])
        repository.insert_events(conn, result.events)
        window = max(event.ts for event in result.events)
        repository.refresh_rollups(
            conn,
            min(event.ts for event in result.events).replace(minute=0, second=0, microsecond=0),
            window,
        )

    metrics.observe_ingest(result, duration_seconds=time.perf_counter() - started)

    response = ExportTraceServiceResponse()
    if result.events_unpriced:
        # Not a rejection: the spans are stored, we just could not attach money to them.
        # OTLP has no "accepted with warnings", so the count travels in the message.
        response.partial_success.CopyFrom(
            ExportTracePartialSuccess(
                rejected_spans=0,
                error_message=(
                    f"{result.events_unpriced} event(s) stored unpriced; "
                    f"unresolved models: {sorted(result.unpriced_models)}"
                ),
            )
        )
    return _encode(response, content_type)


def _encode(response: ExportTraceServiceResponse, content_type: str | None) -> Response:
    media = (content_type or "").split(";")[0].strip().lower()
    if media in PROTOBUF_CONTENT_TYPES:
        return Response(response.SerializeToString(), media_type="application/x-protobuf")
    # Anything else already failed to decode, so JSON is the only case left.
    return Response(json_format.MessageToJson(response), media_type="application/json")


def _error(code: int, message: str) -> Response:
    return Response(
        json_format.MessageToJson(
            ExportTraceServiceResponse(
                partial_success=ExportTracePartialSuccess(error_message=message)
            )
        ),
        status_code=code,
        media_type="application/json",
    )
