"""Synthetic DeepSeek Harness (dsh) traffic for the tokenomics demo.

Companion to ``simulate.py``, themed around the integration in
``tokenomics.importers.dsh`` rather than a generic SaaS fleet: every event has
``attribution.project == "dsh"`` and ``attribution.feature`` set to the repo the agent
was pointed at, exactly like a real imported session. Unlike ``simulate.py`` this
writes ``UsageEvent``\\ s directly instead of round-tripping through OTLP -- there is no
wire format to exercise here, since dsh events reach tokenomics via the importer's
session-log parser, not a span exporter.

Four workloads, chosen to make the self-host/pay-for-API story visible at once:

* **tokenomics** and **dsharness** run on local Ollama (``deepseek-r1``) -- free,
  and therefore UNPRICED, not zero. This is the bulk of the daily volume.
* **mobile-app** pays for the real DeepSeek API (``deepseek-chat``) -- cheap
  ($0.28/M input, $0.028/M on a cache hit) but real money, with a system prompt
  cache hit rate typical of a coding agent re-sending the same tool schema.
* **infra-scripts** pays for Claude (``claude-sonnet-4-5``) for the occasional call
  that needs it -- 10x the rate, low volume, meaningful share of the bill anyway.

One spike is injected: a retry loop on the paid ``mobile-app`` route (dsh's own
``llm/retry-started`` mechanic, gone wrong on a bad deploy), sized to push both the
org-wide and per-feature budget past every threshold in one afternoon.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import psycopg

from tokenomics.finops.budgets import Budget, Period, create_budget, list_budgets
from tokenomics.models import Attribution, TokenVector, UsageEvent
from tokenomics.pricing.engine import PricingEngine, default_engine
from tokenomics.storage import database, repository

SEED = 20260904

#: Same shape as simulate.py's curve: quiet overnight, busy through the working day.
DIURNAL = (
    0.15, 0.10, 0.08, 0.08, 0.10, 0.20, 0.45, 0.80, 1.20, 1.60, 1.80, 1.70,
    1.40, 1.60, 1.75, 1.65, 1.40, 1.10, 0.85, 0.65, 0.50, 0.40, 0.30, 0.20,
)  # fmt: skip
WEEKEND_FACTOR = 0.35


@dataclass(frozen=True, slots=True)
class Workload:
    feature: str
    provider: str
    model: str
    self_hosted: bool
    sessions_per_day: int
    calls_per_session: tuple[int, int]
    input_range: tuple[int, int]
    output_range: tuple[int, int]
    cache_hit_rate: float = 0.0
    cache_share: float = 0.0
    #: Share of calls made by a subagent rather than the root session.
    subagent_share: float = 0.0


WORKLOADS = (
    Workload(
        feature="tokenomics",
        provider="ollama",
        model="deepseek-r1:8b",
        self_hosted=True,
        sessions_per_day=6,
        calls_per_session=(2, 9),
        input_range=(500, 6_000),
        output_range=(150, 2_200),
        subagent_share=0.20,
    ),
    Workload(
        feature="dsharness",
        provider="ollama",
        model="deepseek-r1:14b",
        self_hosted=True,
        sessions_per_day=3,
        calls_per_session=(2, 7),
        input_range=(800, 8_000),
        output_range=(200, 2_800),
        subagent_share=0.10,
    ),
    Workload(
        feature="mobile-app",
        provider="deepseek",
        model="deepseek-chat",
        self_hosted=False,
        sessions_per_day=10,
        calls_per_session=(4, 15),
        input_range=(5_000, 40_000),
        output_range=(400, 3_000),
        cache_hit_rate=0.65,
        cache_share=0.75,
    ),
    Workload(
        feature="infra-scripts",
        provider="anthropic",
        model="claude-sonnet-4-5",
        self_hosted=False,
        sessions_per_day=2,
        calls_per_session=(2, 6),
        input_range=(2_000, 12_000),
        output_range=(200, 1_200),
        cache_hit_rate=0.3,
        cache_share=0.5,
    ),
)


@dataclass(frozen=True, slots=True)
class Call:
    ts: datetime
    workload: Workload
    session_id: str
    turn: int
    step: int
    delegation_depth: int
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_write: int


def _sessions_at(workload: Workload, moment: datetime, rng: random.Random) -> int:
    weight = DIURNAL[moment.hour] / sum(DIURNAL) * 24
    if moment.weekday() >= 5:
        weight *= WEEKEND_FACTOR
    hourly = workload.sessions_per_day / 24 * weight
    # At these session-per-day volumes hourly is usually well under 1 -- a Gaussian
    # sample truncated by int() rounds down to 0 almost every hour, which is exactly
    # what happened on the first run (three of four workloads vanished entirely).
    # Whole part plus a Bernoulli trial on the remainder keeps E[count] == hourly.
    whole, frac = divmod(hourly, 1)
    return int(whole) + (1 if rng.random() < frac else 0)


def _session_calls(workload: Workload, start: datetime, rng: random.Random) -> Iterator[Call]:
    session_id = f"session-{uuid.UUID(int=rng.getrandbits(128))}"
    turns = rng.randint(*workload.calls_per_session)
    moment = start
    for turn in range(1, turns + 1):
        moment += timedelta(seconds=rng.uniform(20, 240))
        input_tokens = rng.randint(*workload.input_range)
        output_tokens = rng.randint(*workload.output_range)

        cache_read = 0
        if workload.cache_hit_rate and rng.random() < workload.cache_hit_rate:
            cache_read = int(input_tokens * workload.cache_share)

        depth = 1 if rng.random() < workload.subagent_share else 0

        yield Call(
            ts=moment,
            workload=workload,
            session_id=session_id,
            turn=turn,
            step=1,
            delegation_depth=depth,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read=cache_read,
            cache_write=0,
        )


def generate(*, days: int = 21, end: datetime | None = None, seed: int = SEED) -> list[Call]:
    """Simulate ``days`` of dsh traffic ending at ``end`` (default: now, on the hour)."""
    rng = random.Random(seed)
    end = (end or datetime.now(UTC)).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    calls: list[Call] = []
    moment = start
    while moment < end:
        for workload in WORKLOADS:
            for _ in range(_sessions_at(workload, moment, rng)):
                session_start = moment + timedelta(seconds=rng.uniform(0, 3600))
                calls.extend(_session_calls(workload, session_start, rng))
        moment += timedelta(hours=1)

    calls.extend(_retry_spike(end, rng))
    return sorted(calls, key=lambda call: call.ts)


def _retry_spike(end: datetime, rng: random.Random) -> list[Call]:
    """A bad deploy sends `mobile-app` into a retry loop against the paid API.

    dsh's own `llm/retry-started` mechanic: each failed attempt is billed as its own
    call. This is deliberately *not* extra sessions -- it is one session, many turns,
    on the workload that costs real money, so both the org-wide and per-feature budget
    feel it the same afternoon.
    """
    workload = next(w for w in WORKLOADS if w.feature == "mobile-app")
    window = (end - timedelta(days=2)).replace(hour=15, minute=0, second=0, microsecond=0)
    session_id = f"session-{uuid.UUID(int=rng.getrandbits(128))}"
    calls: list[Call] = []
    for turn in range(1, 1201):
        calls.append(
            Call(
                ts=window + timedelta(seconds=turn * rng.uniform(4, 9)),
                workload=workload,
                session_id=session_id,
                turn=turn,
                step=1,
                delegation_depth=0,
                input_tokens=rng.randint(25_000, 38_000),
                output_tokens=rng.randint(300, 900),  # truncated retries: little output
                cache_read=0,  # each retry mutates the prompt enough to miss
                cache_write=0,
            )
        )
    return calls


def to_event(call: Call, engine: PricingEngine) -> UsageEvent:
    w = call.workload
    tokens = TokenVector(
        input=call.input_tokens + call.cache_read + call.cache_write,
        output=call.output_tokens,
        cache_read=call.cache_read,
        cache_write=call.cache_write,
    )
    tags = {
        "dsh.turn": str(call.turn),
        "dsh.step": str(call.step),
        "dsh.response_id": f"resp-{uuid.uuid4().hex[:12]}",
    }
    if call.delegation_depth:
        tags["dsh.delegation_depth"] = str(call.delegation_depth)

    event = UsageEvent(
        trace_id=uuid.uuid5(uuid.NAMESPACE_URL, call.session_id).hex,
        span_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{call.session_id}:{call.turn}:{call.step}").hex[
            :16
        ],
        ts=call.ts,
        provider=w.provider,
        request_model=w.model,
        response_model=w.model,
        operation="chat",
        tokens=tokens,
        attribution=Attribution(
            project="dsh", feature=w.feature, subject_id=call.session_id, tags=tags
        ),
    )
    return engine.price(event).event


def _ensure_budget(conn: psycopg.Connection, budget: Budget) -> None:
    """Create ``budget`` unless one with the same name already exists (rerun-safe)."""
    if any(b.name == budget.name for b in list_budgets(conn, enabled_only=False)):
        return
    create_budget(conn, budget)


def main() -> None:
    engine = default_engine()
    calls = generate()
    events = [to_event(call, engine) for call in calls]

    priced = sum(1 for e in events if e.cost is not None)
    total_usd = sum((e.cost.total_usd for e in events if e.cost), Decimal(0))

    with database.connection() as conn:
        database.ensure_partitions(conn, [e.ts for e in events])
        written = repository.insert_events(conn, events)
        repository.refresh_rollups(
            conn,
            min(e.ts for e in events).replace(minute=0, second=0, microsecond=0),
            max(e.ts for e in events) + timedelta(hours=1),
        )
        _ensure_budget(
            conn,
            Budget(
                name="dsh org-wide",
                amount_usd=Decimal("25"),
                scope={"project": "dsh"},
                period=Period.MONTHLY,
            ),
        )
        _ensure_budget(
            conn,
            Budget(
                name="mobile-app guardrail",
                amount_usd=Decimal("12"),
                scope={"project": "dsh", "feature": "mobile-app"},
                period=Period.MONTHLY,
            ),
        )

    by_feature: dict[str, int] = {}
    for call in calls:
        by_feature[call.workload.feature] = by_feature.get(call.workload.feature, 0) + 1

    print(f"generated {len(events)} events across {len(by_feature)} features:")
    for feature, count in sorted(by_feature.items(), key=lambda kv: -kv[1]):
        print(f"  {feature}: {count}")
    print(f"written: {written}  priced: {priced}/{len(events)}  total: ${total_usd:.2f}")
    print(
        'budgets: "dsh org-wide" ($25/mo, project=dsh), '
        '"mobile-app guardrail" ($12/mo, feature=mobile-app)'
    )


if __name__ == "__main__":
    main()
