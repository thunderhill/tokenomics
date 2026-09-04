"""Import DeepSeek Harness (``dsh``) session logs as :class:`UsageEvent`\\ s.

``dsh`` persists one JSON-Lines-per-event session log per run, zstd-compressed as
``session.jsonl.zstd`` under ``$DSH_HOME/sessions/``. This module reads that durable log
directly rather than relying on dsh's own telemetry, for two reasons found while wiring this up:

* dsh's shipped OTel exporter (``dsh-session-telemetry-otel``) emits the **Logs** signal, not
  the Traces signal this project ingests at ``/v1/traces`` -- and its delivery is documented
  best-effort / at-most-once with no durable outbox. The session log is complete and replayable
  where that is not, and reading it needs no plugin inside dsh.
* The obvious read of dsh's own ``TokenUsage`` type -- that usage rides on a ``step/end`` event
  -- does not match what actually lands on disk. A real captured session shows ``step/end`` is
  always just ``{turn, step}``, on both errored and completed turns. Usage instead arrives as an
  ``assistant/chunk`` event with ``chunk.type == "usage"``, followed (before the next
  ``request/header``) by a ``chunk.type == "finish"`` event whose ``replayState.response`` names
  the model and provider that actually served the call -- dsh's analogue of
  ``UsageEvent.response_model``, distinct from what ``request/header`` asked for.

See ``tests/fixtures/dsh_session.jsonl`` for a trimmed real capture (Ollama-backed
``deepseek-r1:8b``) exercising this exact shape.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import zstandard

from tokenomics.models import Attribution, TokenVector, UsageEvent
from tokenomics.pricing.engine import PricingEngine
from tokenomics.pricing.resolver import Method


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the parsed JSON records of one dsh session log.

    Transparently decompresses ``.jsonl.zstd``; a plain ``.jsonl`` (as committed for tests) is
    read as-is. dsh writes its session log incrementally as the session runs, so the frame
    carries no content-size header -- ``ZstdDecompressor.decompress()`` needs one to
    preallocate its output buffer and raises ``could not determine content size in frame
    header`` on a real file. ``stream_reader`` decompresses without that requirement.
    """
    if path.suffix == ".zstd":
        with path.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as reader:
            text = reader.read().decode("utf-8")
    else:
        text = path.read_text("utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            yield json.loads(stripped)


def _hex_id(*parts: str, length: int) -> str:
    """A stable id derived from ``parts``, so re-importing the same file cannot double-bill."""
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return digest[:length]


def _trace_id(session_id: str) -> str:
    """One dsh session is one trace -- every call inside it shares this id."""
    return _hex_id("dsh-session", session_id, length=32)


def _span_id(session_id: str, turn: int, step: int, attempt: int) -> str:
    """One id per ``(turn, step, attempt)``; ``attempt`` disambiguates a retried call."""
    return _hex_id("dsh-call", session_id, str(turn), str(step), str(attempt), length=16)


def _epoch_ms(value: Any) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC)


def _project_name(cwd: str | None) -> str | None:
    """The last path segment of the session's workspace, e.g. ``/mnt/tokenomics`` -> ``tokenomics``.

    Used as the ``feature`` attribution dimension: ``project`` is fixed to ``"dsh"`` here (this
    is all agent spend), so which repository the agent was pointed at is the interesting axis to
    slice by -- and it is more reliably present than dsh's own ``agentPreset``, which a headless
    session (our primary source) does not set at all.
    """
    if not cwd:
        return None
    name = cwd.rstrip("/").rsplit("/", 1)[-1]
    return name or None


def parse_session(records: Iterable[dict[str, Any]]) -> Iterator[UsageEvent]:
    """Convert one session's records into :class:`UsageEvent`\\ s.

    Walks the log once, in order. A ``request/header`` sets the *requested* provider/model for
    every call until the next one; a paired ``usage``/``finish`` chunk closes out one call with
    the tokens actually billed and the model that actually served it. A ``(turn, step)`` that
    never produces a ``usage`` chunk -- aborted before completion, or errored, as
    ``MISSING_CREDENTIAL`` is in practice -- yields nothing; there is no usage to price, so
    nothing is billed as a guess.
    """
    session_id: str | None = None
    cwd: str | None = None
    delegation_depth = 0

    request_provider: str | None = None
    request_model: str | None = None
    request_time_ms: int | None = None

    pending: dict[str, Any] | None = None
    attempts: dict[tuple[int, int], int] = {}

    for record in records:
        rtype = record.get("type")
        data = record.get("data") or {}

        if rtype == "session":
            session_id = record.get("id")
            cwd = record.get("cwd")
            delegation_depth = record.get("delegationDepth", 0)
            continue

        if rtype == "request/header":
            config = (data.get("header") or {}).get("config") or {}
            request_provider = config.get("provider")
            request_model = config.get("model")
            request_time_ms = record.get("time")
            continue

        if rtype != "assistant/chunk":
            continue
        chunk = data.get("chunk") or {}
        chunk_type = chunk.get("type")

        if chunk_type == "usage":
            pending = {
                "turn": data.get("turn"),
                "step": data.get("step"),
                "time_ms": record.get("time"),
                "usage": chunk.get("usage") or {},
            }
            continue

        if chunk_type != "finish" or pending is None:
            continue

        turn, step = pending["turn"], pending["step"]
        key = (turn, step)
        attempt = attempts.get(key, 0)
        attempts[key] = attempt + 1

        response = (chunk.get("replayState") or {}).get("response") or {}
        usage = pending["usage"]
        # dsh reports disjoint counts (`inputTokens` is uncached-only), the same asymmetry
        # Anthropic has -- fold cache back into input to reach semconv's inclusive form.
        uncached_input = int(usage.get("inputTokens") or 0)
        cache_read = int(usage.get("cacheReadTokens") or 0)
        cache_write = int(usage.get("cacheWriteTokens") or 0)

        tags = {"dsh.turn": str(turn), "dsh.step": str(step)}
        if delegation_depth:
            tags["dsh.delegation_depth"] = str(delegation_depth)
        if response_id := response.get("responseId"):
            tags["dsh.response_id"] = response_id

        yield UsageEvent(
            trace_id=_trace_id(session_id or "unknown-session"),
            span_id=_span_id(session_id or "unknown-session", turn, step, attempt),
            ts=_epoch_ms(pending["time_ms"]),
            duration_ms=(
                float(pending["time_ms"] - request_time_ms) if request_time_ms is not None else None
            ),
            provider=response.get("provider") or request_provider,
            request_model=request_model,
            response_model=response.get("model"),
            operation="chat",
            tokens=TokenVector(
                input=uncached_input + cache_read + cache_write,
                output=int(usage.get("outputTokens") or 0),
                cache_read=cache_read,
                cache_write=cache_write,
                reasoning=int(usage.get("reasoningTokens") or 0),
            ),
            attribution=Attribution(
                project="dsh",
                feature=_project_name(cwd),
                subject_id=session_id,
                tags=tags,
            ),
        )
        pending = None


def parse_session_file(path: Path) -> Iterator[UsageEvent]:
    """Parse one session log file (``.jsonl`` or ``.jsonl.zstd``) end to end."""
    yield from parse_session(read_records(path))


def find_session_files(root: Path) -> list[Path]:
    """Every session log under a dsh sessions root, compressed or plain, oldest first."""
    files = [*root.glob("**/session.jsonl.zstd"), *root.glob("**/session.jsonl")]
    return sorted(files, key=lambda p: p.stat().st_mtime)


@dataclass(slots=True)
class ImportResult:
    """What happened importing one or more dsh session logs.

    Mirrors :class:`tokenomics.ingest.pipeline.IngestResult` in shape; kept separate because
    "spans received" is an OTLP concept this importer has no equivalent of.
    """

    sessions_read: int = 0
    events_accepted: int = 0
    events_unpriced: int = 0
    unpriced_models: dict[str, int] = field(default_factory=dict)
    events: list[UsageEvent] = field(default_factory=list)


def import_paths(paths: Iterable[Path], engine: PricingEngine) -> ImportResult:
    """Parse and price every session log in ``paths``. Pure -- no database I/O."""
    result = ImportResult()
    for path in paths:
        result.sessions_read += 1
        for event in parse_session_file(path):
            priced = engine.price(event)
            result.events.append(priced.event)
            result.events_accepted += 1
            if priced.resolution.method is Method.UNPRICED:
                result.events_unpriced += 1
                model = event.billing_model or "<unknown>"
                result.unpriced_models[model] = result.unpriced_models.get(model, 0) + 1
    return result
