"""Connection management, migrations and partition maintenance."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg_pool import ConnectionPool

DEFAULT_DSN = "postgresql://tokenomics:tokenomics@localhost:5432/tokenomics"
_MIGRATIONS = Path(__file__).resolve().parent / "migrations"

_pool: ConnectionPool | None = None


def dsn() -> str:
    return os.environ.get("TOKENOMICS_DATABASE_URL", DEFAULT_DSN)


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(dsn(), min_size=1, max_size=10, open=True)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    with pool().connection() as conn:
        yield conn


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply any migrations not yet recorded. Returns the names applied."""
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS schema_migration ("
            " name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        cur.execute("SELECT name FROM schema_migration")
        applied = {row[0] for row in cur.fetchall()}

    performed: list[str] = []
    for path in sorted(_MIGRATIONS.glob("*.sql")):
        if path.name in applied:
            continue
        with conn.cursor() as cur:
            cur.execute(path.read_text())
            cur.execute("INSERT INTO schema_migration (name) VALUES (%s)", (path.name,))
        performed.append(path.name)
    conn.commit()
    return performed


# --------------------------------------------------------------------------- partitions

_known_partitions: set[str] = set()


def _month_bounds(moment: datetime | date) -> tuple[date, date]:
    start = date(moment.year, moment.month, 1)
    end = date(start.year + 1, 1, 1) if start.month == 12 else date(start.year, start.month + 1, 1)
    return start, end


def ensure_partition(conn: psycopg.Connection, moment: datetime | date) -> str:
    """Create the monthly partition covering ``moment`` if it does not exist.

    There is deliberately no DEFAULT partition: one would silently absorb rows for
    months we have not provisioned, and would then block attaching the real partition
    later (Postgres refuses if conflicting rows exist in the default).
    """
    start, end = _month_bounds(moment)
    name = f"usage_event_{start:%Y%m}"
    if name in _known_partitions:
        return name

    # Postgres forbids bind parameters in DDL, so the bounds are composed as literals.
    # They are `datetime.date` objects we derived ourselves, never user input.
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {name} PARTITION OF usage_event "
                "FOR VALUES FROM ({start}) TO ({end})"
            ).format(
                name=sql.Identifier(name),
                start=sql.Literal(start.isoformat()),
                end=sql.Literal(end.isoformat()),
            )
        )
    _known_partitions.add(name)
    return name


def ensure_partitions(conn: psycopg.Connection, moments: list[datetime | date]) -> set[str]:
    return {ensure_partition(conn, moment) for moment in moments}


def reset_partition_cache() -> None:
    """Tests drop and recreate the schema; the in-process cache must follow."""
    _known_partitions.clear()
