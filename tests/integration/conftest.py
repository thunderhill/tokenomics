"""Integration fixtures. Skipped entirely unless TOKENOMICS_TEST_PG_URL is set.

Each test runs inside a transaction that is rolled back, so the suite is order-
independent and leaves no residue -- except partition DDL, which is created once per
session because Postgres cannot attach a partition inside a rolled-back transaction
without re-doing it for every test.

The database named by TOKENOMICS_TEST_PG_URL is treated as **disposable**: the session
fixture truncates the event, rollup and budget tables before the suite runs, so tests
assert on absolute totals rather than deltas. Point the variable at a scratch database
(``docker compose up postgres`` provides one), never at anything you care about.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import psycopg
import pytest

from tokenomics.models import Attribution, ServiceTier, TokenVector, UsageEvent
from tokenomics.storage import database, repository
from tokenomics.storage.queries import Filters

pytestmark = pytest.mark.integration

TEST_URL_ENV = "TOKENOMICS_TEST_PG_URL"

#: Anchored once per run so every fixture lands in the same partitions.
NOW = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)


@pytest.fixture(scope="session")
def pg_url() -> str:
    url = os.environ.get(TEST_URL_ENV)
    if not url:
        pytest.skip(f"{TEST_URL_ENV} is not set")
    return url


@pytest.fixture(scope="session")
def _schema(pg_url: str) -> Iterator[None]:
    with psycopg.connect(pg_url) as conn:
        database.migrate(conn)
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE usage_event, usage_rollup_hourly, budget, anomaly "
                "RESTART IDENTITY CASCADE"
            )
        # Cover the window the fixtures write into, plus the month either side.
        database.ensure_partitions(
            conn,
            [
                NOW - timedelta(days=45),
                NOW - timedelta(days=15),
                NOW,
                NOW + timedelta(days=31),
            ],
        )
        conn.commit()
    yield
    database.reset_partition_cache()


@pytest.fixture
def conn(pg_url: str, _schema: None) -> Iterator[psycopg.Connection]:
    with psycopg.connect(pg_url) as connection:
        yield connection
        connection.rollback()


@pytest.fixture(scope="session")
def now() -> datetime:
    return NOW


@pytest.fixture
def window():
    """Factory for a Filters covering the range the fixtures write into.

    Exposed as a fixture rather than an importable helper because ``tests`` is not a
    package -- a sibling test module cannot import from another one.
    """

    def build(**kwargs) -> Filters:
        return Filters(since=NOW - timedelta(days=2), until=NOW + timedelta(days=1), **kwargs)

    return build


@pytest.fixture
def make_event(engine):
    """Build a priced event. ``n`` makes the span id unique."""

    def build(
        *,
        n,
        model="gpt-4o",
        project="checkout",
        feature="search",
        subject="u-1",
        ts=None,
        tokens=None,
        tags=None,
    ) -> UsageEvent:
        event = UsageEvent(
            trace_id=f"trace-{n:04d}",
            span_id=f"span-{n:04d}",
            ts=ts or NOW,
            provider="openai",
            request_model=model,
            response_model=model,
            service_tier=ServiceTier.STANDARD,
            tokens=tokens or TokenVector(input=1_000, output=200),
            attribution=Attribution(
                project=project,
                feature=feature,
                environment="prod",
                subject_id=subject,
                tags=tags or {},
            ),
        )
        return engine.price(event).event

    return build


@pytest.fixture
def seed(conn, make_event):
    """Insert one event and return what it cost."""

    def insert(**kwargs) -> Decimal:
        event = make_event(**kwargs)
        repository.insert_events(conn, [event])
        return event.cost.total_usd if event.cost else Decimal(0)

    return insert
