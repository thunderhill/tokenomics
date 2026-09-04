"""Budgets: scoped spend limits with fire-once thresholds and signed webhooks.

Three details make the difference between a budget that people trust and one they
mute after a week:

**Fire-once.** A budget is evaluated on a schedule, so a naive implementation alerts
every single run once spend crosses 80%. The ``budget_alert`` table's
``UNIQUE (budget_id, period_start, threshold)`` constraint *is* the state machine:
inserting with ``ON CONFLICT DO NOTHING ... RETURNING`` yields rows only the first
time each threshold is crossed in each period.

**Blind spots are reported, not hidden.** ``sum(cost_usd)`` silently skips unpriced
events, so a budget sitting on a model we could not resolve would read as *under*
budget. Every status carries the unpriced count, and the webhook payload carries it
too.

**Signed webhooks.** Payloads are signed HMAC-SHA256 over ``timestamp.body`` so a
receiver can verify origin and reject replays. There is no SaaS involved: the
webhook goes wherever you point it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID

import httpx
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from tokenomics.storage.queries import Filters, spend_breakdown

#: Scope keys that map onto an indexed attribution column. Anything else is rejected
#: rather than silently ignored -- a budget scoped to a typo would happily watch the
#: whole estate and never fire.
SCOPE_FIELDS: frozenset[str] = frozenset(
    {
        "project",
        "feature",
        "environment",
        "subject_id",
        "prompt_version",
        "model_key",
        "provider",
    }
)

DEFAULT_THRESHOLDS: tuple[Decimal, ...] = (Decimal("0.5"), Decimal("0.8"), Decimal("1.0"))

#: Webhook delivery: total attempts and the base for exponential backoff.
WEBHOOK_ATTEMPTS = 3
WEBHOOK_BACKOFF_SECONDS = 0.5
WEBHOOK_TIMEOUT_SECONDS = 10.0

SIGNATURE_HEADER = "X-Tokenomics-Signature"
TIMESTAMP_HEADER = "X-Tokenomics-Timestamp"


class InvalidScopeError(ValueError):
    """A budget scope references something that is not an attribution dimension."""


class Period(StrEnum):
    MONTHLY = "monthly"
    ROLLING = "rolling"


@dataclass(frozen=True, slots=True)
class Budget:
    """A spend limit over a scope and a period."""

    name: str
    amount_usd: Decimal
    scope: Mapping[str, Any] = field(default_factory=dict)
    period: Period = Period.MONTHLY
    rolling_days: int | None = None
    thresholds: tuple[Decimal, ...] = DEFAULT_THRESHOLDS
    webhook_url: str | None = None
    webhook_secret: str | None = None
    enabled: bool = True
    id: UUID | None = None

    def __post_init__(self) -> None:
        if self.amount_usd <= 0:
            raise ValueError("budget amount must be positive")
        if self.period is Period.ROLLING and not self.rolling_days:
            raise ValueError("a rolling budget needs rolling_days")
        validate_scope(self.scope)


@dataclass(frozen=True, slots=True)
class BudgetStatus:
    """Where a budget stands right now."""

    budget: Budget
    period_start: date
    since: datetime
    until: datetime
    spend_usd: Decimal
    requests: int
    unpriced_events: int
    forecast_usd: Decimal | None = None

    @property
    def amount_usd(self) -> Decimal:
        return self.budget.amount_usd

    @property
    def remaining_usd(self) -> Decimal:
        return self.amount_usd - self.spend_usd

    @property
    def utilization(self) -> Decimal:
        return self.spend_usd / self.amount_usd

    @property
    def projected_utilization(self) -> Decimal | None:
        if self.forecast_usd is None:
            return None
        return self.forecast_usd / self.amount_usd

    @property
    def has_blind_spot(self) -> bool:
        """True when some events in scope could not be priced, so spend is understated."""
        return self.unpriced_events > 0

    def crossed(self) -> tuple[Decimal, ...]:
        """Thresholds this period's spend has reached, lowest first."""
        return tuple(sorted(t for t in self.budget.thresholds if self.utilization >= t))


@dataclass(frozen=True, slots=True)
class Alert:
    """A threshold crossing that had not fired before."""

    id: int
    budget: Budget
    period_start: date
    threshold: Decimal
    spend_usd: Decimal
    amount_usd: Decimal
    fired_at: datetime

    @property
    def utilization(self) -> Decimal:
        return self.spend_usd / self.amount_usd


def validate_scope(scope: Mapping[str, Any]) -> None:
    """Reject scope keys that are not attribution dimensions."""
    for key in scope:
        if key == "tags":
            if not isinstance(scope[key], Mapping):
                raise InvalidScopeError("scope 'tags' must be an object of tag -> value")
            continue
        if key not in SCOPE_FIELDS:
            raise InvalidScopeError(
                f"unknown scope field {key!r}; allowed: {sorted(SCOPE_FIELDS)} or 'tags'"
            )


def scope_filters(scope: Mapping[str, Any], since: datetime, until: datetime) -> Filters:
    """Turn a stored scope document into a bound-parameter query filter."""
    validate_scope(scope)
    kwargs: dict[str, Any] = {"since": since, "until": until}
    for key, value in scope.items():
        if key == "tags":
            kwargs["tags"] = {str(k): str(v) for k, v in value.items()}
        elif isinstance(value, str):
            kwargs[key] = (value,)
        else:
            kwargs[key] = tuple(str(v) for v in value)
    return Filters(**kwargs)


def period_bounds(budget: Budget, now: datetime) -> tuple[date, datetime, datetime]:
    """Return ``(period_start, since, until)`` for the period containing ``now``.

    For a monthly budget the period start is the calendar month, so a threshold fires
    at most once per month. A rolling window slides continuously and has no natural
    anchor, so we key its state on the *as-of day*: a rolling budget can alert at most
    once per threshold per day, and can alert again tomorrow if it is still over.
    """
    now = now.astimezone(UTC) if now.tzinfo else now.replace(tzinfo=UTC)
    if budget.period is Period.MONTHLY:
        since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        until = _add_month(since)
        return since.date(), since, until

    days = budget.rolling_days or 30
    return now.date(), now - timedelta(days=days), now


def _add_month(moment: datetime) -> datetime:
    if moment.month == 12:
        return moment.replace(year=moment.year + 1, month=1)
    return moment.replace(month=moment.month + 1)


def evaluate(
    conn: psycopg.Connection,
    budget: Budget,
    *,
    now: datetime | None = None,
    forecast_usd: Decimal | None = None,
) -> BudgetStatus:
    """Measure spend in a budget's scope over its current period."""
    now = now or datetime.now(UTC)
    period_start, since, until = period_bounds(budget, now)
    rows = spend_breakdown(conn, filters=scope_filters(budget.scope, since, until), group_by=[])
    row: dict[str, Any] = rows[0] if rows else {}
    return BudgetStatus(
        budget=budget,
        period_start=period_start,
        since=since,
        until=until,
        spend_usd=Decimal(row.get("cost_usd") or 0),
        requests=int(row.get("requests") or 0),
        unpriced_events=int(row.get("unpriced_events") or 0),
        forecast_usd=forecast_usd,
    )


def fire_alerts(conn: psycopg.Connection, status: BudgetStatus) -> list[Alert]:
    """Record newly crossed thresholds. Already-fired thresholds return nothing.

    The UNIQUE constraint does the de-duplication, so this is safe to call from several
    schedulers at once: exactly one of them gets the row back.
    """
    if status.budget.id is None:
        raise ValueError("budget must be persisted before alerts can fire")
    crossed = status.crossed()
    if not crossed:
        return []

    alerts: list[Alert] = []
    with conn.cursor(row_factory=dict_row) as cur:
        for threshold in crossed:
            cur.execute(
                """
                INSERT INTO budget_alert
                    (budget_id, period_start, threshold, spend_usd, amount_usd)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (budget_id, period_start, threshold) DO NOTHING
                RETURNING id, fired_at
                """,
                (
                    status.budget.id,
                    status.period_start,
                    threshold,
                    status.spend_usd,
                    status.amount_usd,
                ),
            )
            inserted = cur.fetchone()
            if inserted is None:
                continue  # already fired this period
            alerts.append(
                Alert(
                    id=int(inserted["id"]),
                    budget=status.budget,
                    period_start=status.period_start,
                    threshold=threshold,
                    spend_usd=status.spend_usd,
                    amount_usd=status.amount_usd,
                    fired_at=inserted["fired_at"],
                )
            )
    return alerts


def alert_payload(alert: Alert, status: BudgetStatus | None = None) -> dict[str, Any]:
    """The JSON body posted to a budget webhook."""
    payload: dict[str, Any] = {
        "type": "budget.threshold_crossed",
        "alert_id": alert.id,
        "budget": {
            "id": str(alert.budget.id),
            "name": alert.budget.name,
            "scope": dict(alert.budget.scope),
            "period": str(alert.budget.period),
        },
        "period_start": alert.period_start.isoformat(),
        "threshold": float(alert.threshold),
        "spend_usd": str(alert.spend_usd),
        "amount_usd": str(alert.amount_usd),
        "utilization": float(alert.utilization),
        "fired_at": alert.fired_at.isoformat(),
    }
    if status is not None:
        payload["requests"] = status.requests
        # Surfaced deliberately: unpriced events mean the real number is higher.
        payload["unpriced_events"] = status.unpriced_events
        if status.forecast_usd is not None:
            payload["forecast_usd"] = str(status.forecast_usd)
    return payload


def sign(secret: str, body: bytes, timestamp: int) -> str:
    """Stripe-style signature over ``timestamp.body`` so replays can be rejected."""
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256)
    return f"t={timestamp},v1={mac.hexdigest()}"


def verify(secret: str, body: bytes, header: str, *, tolerance_seconds: int = 300) -> bool:
    """Receiver-side counterpart to :func:`sign`. Provided so tests and docs agree."""
    parts = dict(p.split("=", 1) for p in header.split(",") if "=" in p)
    try:
        timestamp = int(parts["t"])
    except (KeyError, ValueError):
        return False
    if abs(time.time() - timestamp) > tolerance_seconds:
        return False
    expected = sign(secret, body, timestamp)
    return hmac.compare_digest(expected, header)


def deliver(
    alert: Alert,
    payload: Mapping[str, Any],
    *,
    client: httpx.Client | None = None,
    attempts: int = WEBHOOK_ATTEMPTS,
    sleep: Any = time.sleep,
) -> tuple[bool, str | None]:
    """POST a signed alert, retrying transient failures. Returns ``(ok, error)``."""
    url = alert.budget.webhook_url
    if not url:
        return False, "no webhook configured"

    body = json.dumps(payload, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json"}
    if alert.budget.webhook_secret:
        timestamp = int(time.time())
        headers[SIGNATURE_HEADER] = sign(alert.budget.webhook_secret, body, timestamp)
        headers[TIMESTAMP_HEADER] = str(timestamp)

    owned = client is None
    http = client or httpx.Client(timeout=WEBHOOK_TIMEOUT_SECONDS)
    error: str | None = None
    try:
        for attempt in range(attempts):
            try:
                response = http.post(url, content=body, headers=headers)
            except httpx.HTTPError as exc:
                error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code < 400:
                    return True, None
                error = f"HTTP {response.status_code}"
                # 4xx is a permanent rejection: retrying just repeats the mistake.
                if response.status_code < 500:
                    break
            if attempt < attempts - 1:
                sleep(WEBHOOK_BACKOFF_SECONDS * (2**attempt))
    finally:
        if owned:
            http.close()
    return False, error


def mark_delivered(
    conn: psycopg.Connection, alert_id: int, *, ok: bool, error: str | None = None
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE budget_alert SET delivered = %s, delivery_error = %s WHERE id = %s",
            (ok, error, alert_id),
        )


# --- persistence -----------------------------------------------------------------

_BUDGET_COLUMNS = (
    "id, name, scope, amount_usd, period, rolling_days, thresholds, "
    "webhook_url, webhook_secret, enabled"
)


def _budget_from_row(row: Mapping[str, Any]) -> Budget:
    return Budget(
        id=row["id"],
        name=row["name"],
        scope=row["scope"] or {},
        amount_usd=Decimal(row["amount_usd"]),
        period=Period(row["period"]),
        rolling_days=row["rolling_days"],
        thresholds=tuple(Decimal(t) for t in row["thresholds"]),
        webhook_url=row["webhook_url"],
        webhook_secret=row["webhook_secret"],
        enabled=row["enabled"],
    )


def create_budget(conn: psycopg.Connection, budget: Budget) -> Budget:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            INSERT INTO budget
                (name, scope, amount_usd, period, rolling_days, thresholds,
                 webhook_url, webhook_secret, enabled)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING {_BUDGET_COLUMNS}
            """,
            (
                budget.name,
                Jsonb(dict(budget.scope)),
                budget.amount_usd,
                str(budget.period),
                budget.rolling_days,
                list(budget.thresholds),
                budget.webhook_url,
                budget.webhook_secret,
                budget.enabled,
            ),
        )
        row = cur.fetchone()
    assert row is not None  # RETURNING on a successful INSERT always yields a row
    return _budget_from_row(row)


def list_budgets(conn: psycopg.Connection, *, enabled_only: bool = True) -> list[Budget]:
    clause = "WHERE enabled" if enabled_only else ""
    with conn.cursor(row_factory=dict_row) as cur:
        # `clause` and `_BUDGET_COLUMNS` are literals chosen here, never request input.
        cur.execute(f"SELECT {_BUDGET_COLUMNS} FROM budget {clause} ORDER BY name")
        return [_budget_from_row(row) for row in cur.fetchall()]


def get_budget(conn: psycopg.Connection, budget_id: UUID) -> Budget | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT {_BUDGET_COLUMNS} FROM budget WHERE id = %s", (budget_id,))
        row = cur.fetchone()
    return _budget_from_row(row) if row else None


def delete_budget(conn: psycopg.Connection, budget_id: UUID) -> bool:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM budget WHERE id = %s", (budget_id,))
        return cur.rowcount > 0


def recent_alerts(
    conn: psycopg.Connection, *, budget_id: UUID | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    # `where` is one of two literals; the id itself is always a bound parameter.
    where = "WHERE a.budget_id = %(budget_id)s" if budget_id else ""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT a.id, a.budget_id, b.name AS budget_name, a.period_start, a.threshold,
                   a.spend_usd, a.amount_usd, a.fired_at, a.delivered, a.delivery_error
            FROM budget_alert a JOIN budget b ON b.id = a.budget_id
            {where}
            ORDER BY a.fired_at DESC LIMIT %(limit)s
            """,
            {"budget_id": budget_id, "limit": limit},
        )
        return list(cur.fetchall())


def run_budget_cycle(
    conn: psycopg.Connection,
    *,
    now: datetime | None = None,
    client: httpx.Client | None = None,
    budgets: Sequence[Budget] | None = None,
) -> list[Alert]:
    """Evaluate every enabled budget, fire new thresholds, and deliver the webhooks."""
    fired: list[Alert] = []
    for budget in budgets if budgets is not None else list_budgets(conn):
        status = evaluate(conn, budget, now=now)
        for alert in fire_alerts(conn, status):
            fired.append(alert)
            if budget.webhook_url:
                ok, error = deliver(alert, alert_payload(alert, status), client=client)
                mark_delivered(conn, alert.id, ok=ok, error=error)
    return fired
