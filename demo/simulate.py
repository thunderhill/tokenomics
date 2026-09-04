"""Traffic generator for the Tokenomics demo.

Builds real OTLP/protobuf payloads -- the same bytes an OTel exporter would send -- for
six projects with different models, cost profiles and daily rhythms, then posts them to
a running Tokenomics API.

Every shape here exists to exercise something that is easy to get wrong, and each one
is a finding the tool should surface rather than decoration:

* **checkout** is high-volume and cheap (gpt-4o-mini, with a slice of gpt-4o).
* **support** is cache-heavy Anthropic traffic. Most of its input tokens are cache
  reads at 0.1x, so a system that sums components instead of partitioning them will
  overstate this project by roughly an order of magnitude.
* **insights** runs long-context Gemini calls with reasoning tokens -- inclusive inside
  the output total, and *not* billed apart, because gemini-2.5-pro quotes no separate
  reasoning rate. Its reasoning volume is real; its reasoning line item is zero.
* **research** is the other half of that story: qwen-plus does price reasoning apart,
  at 4.00/M against a 1.20/M output rate. Thinking costs 3.3x answering, and only a
  tool that splits the two can tell you so.
* **assistant** thrashes its prompt cache -- it writes far more often than it reads.
  A cache write is ~1.25x the input rate and a read ~0.1x, so this project pays a
  premium for a cache that is not earning it back. Scored on savings alone it would
  look like a success; scored net it is a loss.
* **platform** calls an internal model gateway that no price list knows. Its cost is
  stored NULL, never zero, so the spend it represents shows up as a stated blind spot
  instead of quietly making the totals look better than they are.

Traffic follows a business-hours diurnal curve and a weekend dip, and one deliberate
spike is injected so anomaly detection has something real to find.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

SEED = 20260821

#: Hour-of-day weights: quiet overnight, busy through the working day.
DIURNAL = (
    0.15,
    0.10,
    0.08,
    0.08,
    0.10,
    0.20,
    0.45,
    0.80,
    1.20,
    1.60,
    1.80,
    1.70,
    1.40,
    1.60,
    1.75,
    1.65,
    1.40,
    1.10,
    0.85,
    0.65,
    0.50,
    0.40,
    0.30,
    0.20,
)
WEEKEND_FACTOR = 0.35


@dataclass(frozen=True, slots=True)
class Project:
    name: str
    provider: str
    model: str
    team: str
    features: tuple[str, ...]
    calls_per_day: int
    input_range: tuple[int, int]
    output_range: tuple[int, int]
    #: Probability a call hits the prompt cache, and the share of input it covers.
    cache_hit_rate: float = 0.0
    cache_share: float = 0.0
    reasoning_share: float = 0.0
    #: Occasional traffic on a second, pricier model.
    secondary_model: str | None = None
    secondary_share: float = 0.0
    subjects: int = 40


PROJECTS = (
    Project(
        name="checkout",
        provider="openai",
        model="gpt-4o-mini",
        team="growth",
        features=("cart-summarizer", "product-qa", "fraud-triage"),
        calls_per_day=900,
        input_range=(700, 2_500),
        output_range=(80, 400),
        secondary_model="gpt-4o",
        secondary_share=0.12,
        subjects=120,
    ),
    Project(
        name="support",
        provider="anthropic",
        model="claude-sonnet-4-5",
        team="cx",
        features=("ticket-triage", "reply-drafter", "kb-search"),
        calls_per_day=420,
        input_range=(6_000, 22_000),
        output_range=(200, 900),
        cache_hit_rate=0.72,
        cache_share=0.88,
        subjects=60,
    ),
    Project(
        name="insights",
        provider="gcp.gemini",
        model="gemini-2.5-pro",
        team="data",
        features=("weekly-digest", "anomaly-explainer", "auto-summarize"),
        calls_per_day=110,
        input_range=(20_000, 140_000),
        output_range=(600, 2_400),
        reasoning_share=0.45,
        subjects=25,
    ),
    # Reasoning priced apart: qwen-plus bills thinking at 4.00/M against a 1.20/M
    # output rate, so the reasoning component carries a share of the bill far larger
    # than its share of the volume. Contrast with `insights`, whose model bundles it.
    Project(
        name="research",
        provider="dashscope",
        model="qwen-plus-latest",
        team="data",
        features=("literature-review", "hypothesis-check", "report-writer"),
        calls_per_day=95,
        input_range=(3_000, 14_000),
        output_range=(1_500, 5_000),
        reasoning_share=0.62,
        subjects=18,
    ),
    # A cache that costs more than it saves: a low hit rate against a large cached
    # prefix means most calls pay the ~1.25x write premium and few collect the ~0.1x
    # read. Reported as a saving alone this looks like a win, which is the point.
    Project(
        name="assistant",
        provider="anthropic",
        model="claude-haiku-4-5",
        team="growth",
        features=("inline-help", "onboarding-bot"),
        calls_per_day=260,
        input_range=(4_000, 15_000),
        output_range=(150, 600),
        cache_hit_rate=0.15,
        cache_share=0.85,
        subjects=80,
    ),
    # Unpriced on purpose: an internal gateway no price list carries. Cost is stored
    # NULL rather than 0, so this traffic is reported as a blind spot instead of
    # silently improving every average it touches.
    Project(
        name="platform",
        provider="acme",
        model="internal-router-v2",
        team="platform",
        features=("embedding-refresh", "batch-classify"),
        calls_per_day=70,
        input_range=(1_000, 4_500),
        output_range=(100, 500),
        cache_hit_rate=0.5,
        cache_share=0.6,
        subjects=12,
    ),
)


@dataclass(frozen=True, slots=True)
class Call:
    """One simulated LLM call, ready to be encoded as a span."""

    ts: datetime
    project: str
    feature: str
    model: str
    provider: str
    subject: str
    prompt_version: str
    team: str
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_write: int
    reasoning: int
    duration_ms: float


def _volume(project: Project, moment: datetime, rng: random.Random) -> int:
    weight = DIURNAL[moment.hour] / sum(DIURNAL) * 24
    if moment.weekday() >= 5:
        weight *= WEEKEND_FACTOR
    hourly = project.calls_per_day / 24 * weight
    return max(0, int(rng.gauss(hourly, hourly * 0.25)))


def _call(project: Project, moment: datetime, rng: random.Random) -> Call:
    model = project.model
    if project.secondary_model and rng.random() < project.secondary_share:
        model = project.secondary_model

    input_tokens = rng.randint(*project.input_range)
    output_tokens = rng.randint(*project.output_range)

    cache_read = cache_write = 0
    if project.cache_hit_rate and rng.random() < project.cache_hit_rate:
        cache_read = int(input_tokens * project.cache_share)
    elif project.cache_hit_rate:
        # A miss writes the prefix into the cache instead of reading it.
        cache_write = int(input_tokens * project.cache_share)

    reasoning = int(output_tokens * project.reasoning_share) if project.reasoning_share else 0

    return Call(
        ts=moment + timedelta(seconds=rng.uniform(0, 3600)),
        project=project.name,
        feature=rng.choice(project.features),
        model=model,
        provider=project.provider,
        subject=f"cust_{rng.randrange(project.subjects):04d}",
        prompt_version=rng.choice(("v3", "v4")),
        team=project.team,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        reasoning=reasoning,
        duration_ms=rng.uniform(300, 4_000),
    )


def generate(
    *, days: int = 45, end: datetime | None = None, seed: int = SEED, spike: bool = True
) -> list[Call]:
    """Simulate ``days`` of traffic ending at ``end`` (default: now, on the hour)."""
    rng = random.Random(seed)
    end = (end or datetime.now(UTC)).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    calls: list[Call] = []
    moment = start
    while moment < end:
        for project in PROJECTS:
            for _ in range(_volume(project, moment, rng)):
                calls.append(_call(project, moment, rng))
        moment += timedelta(hours=1)

    if spike:
        calls.extend(_spike(end, rng))
    return sorted(calls, key=lambda call: call.ts)


def _spike(end: datetime, rng: random.Random) -> list[Call]:
    """A retry loop on an expensive model: the anomaly the demo goes looking for.

    Deliberately *not* a volume spike on the usual model -- the interesting failure is
    a cheap feature quietly switching to an expensive one, which is exactly what
    probable-cause attribution should surface.
    """
    window = end - timedelta(days=2)
    window = window.replace(hour=14)
    calls: list[Call] = []
    for hour in range(2):
        for _ in range(220):
            moment = window + timedelta(hours=hour)
            calls.append(
                Call(
                    ts=moment + timedelta(seconds=rng.uniform(0, 3600)),
                    project="insights",
                    feature="auto-summarize",
                    model="gpt-4o",
                    provider="openai",
                    subject=f"cust_{rng.randrange(25):04d}",
                    prompt_version="v5",
                    team="data",
                    input_tokens=rng.randint(90_000, 120_000),
                    output_tokens=rng.randint(1_500, 3_000),
                    cache_read=0,
                    cache_write=0,
                    reasoning=0,
                    duration_ms=rng.uniform(5_000, 20_000),
                )
            )
    return calls


# --- OTLP encoding -----------------------------------------------------------------


def _attr(key: str, value: object) -> KeyValue:
    if isinstance(value, bool):
        any_value = AnyValue(bool_value=value)
    elif isinstance(value, int):
        any_value = AnyValue(int_value=value)
    elif isinstance(value, float):
        any_value = AnyValue(double_value=value)
    else:
        any_value = AnyValue(string_value=str(value))
    return KeyValue(key=key, value=any_value)


def to_span(call: Call, index: int) -> Span:
    """Encode one call using current GenAI semantic-convention attribute names."""
    start_nanos = int(call.ts.timestamp() * 1_000_000_000)
    attributes = [
        _attr("gen_ai.operation.name", "chat"),
        _attr("gen_ai.provider.name", call.provider),
        _attr("gen_ai.request.model", call.model),
        _attr("gen_ai.response.model", call.model),
        _attr("gen_ai.usage.input_tokens", call.input_tokens),
        _attr("gen_ai.usage.output_tokens", call.output_tokens),
        _attr("tokenomics.project", call.project),
        _attr("tokenomics.feature", call.feature),
        _attr("tokenomics.environment", "prod"),
        _attr("tokenomics.subject_id", call.subject),
        _attr("tokenomics.prompt_version", call.prompt_version),
        _attr("tokenomics.tag.team", call.team),
    ]
    if call.cache_read:
        attributes.append(_attr("gen_ai.usage.cache_read.input_tokens", call.cache_read))
    if call.cache_write:
        attributes.append(_attr("gen_ai.usage.cache_write.input_tokens", call.cache_write))
    if call.reasoning:
        attributes.append(_attr("gen_ai.usage.reasoning.output_tokens", call.reasoning))

    return Span(
        trace_id=index.to_bytes(16, "big"),
        span_id=index.to_bytes(8, "big"),
        name=f"chat {call.model}",
        kind=Span.SPAN_KIND_CLIENT,
        start_time_unix_nano=start_nanos,
        end_time_unix_nano=start_nanos + int(call.duration_ms * 1_000_000),
        attributes=attributes,
    )


def to_otlp(calls: list[Call], *, offset: int = 0) -> bytes:
    """Serialize a batch as an OTLP ExportTraceServiceRequest."""
    request = ExportTraceServiceRequest(
        resource_spans=[
            ResourceSpans(
                resource=Resource(attributes=[_attr("service.name", "tokenomics-demo")]),
                scope_spans=[
                    ScopeSpans(spans=[to_span(call, offset + i) for i, call in enumerate(calls)])
                ],
            )
        ]
    )
    return request.SerializeToString()


def batches(calls: list[Call], size: int = 400) -> Iterator[tuple[int, list[Call]]]:
    for start in range(0, len(calls), size):
        yield start, calls[start : start + size]
