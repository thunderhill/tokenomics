"""Budget periods, the fire-once contract's inputs, and webhook signing."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import httpx
import pytest

from tokenomics.finops import budgets
from tokenomics.finops.budgets import (
    Alert,
    Budget,
    BudgetStatus,
    InvalidScopeError,
    Period,
)


def make_budget(**kwargs: object) -> Budget:
    defaults: dict[str, object] = {
        "name": "checkout",
        "amount_usd": Decimal("100"),
        "scope": {"project": "checkout"},
        "id": uuid4(),
    }
    return Budget(**{**defaults, **kwargs})  # type: ignore[arg-type]


def make_status(spend: str, **kwargs: object) -> BudgetStatus:
    budget = kwargs.pop("budget", None) or make_budget()
    return BudgetStatus(
        budget=budget,  # type: ignore[arg-type]
        period_start=datetime(2026, 8, 1, tzinfo=UTC).date(),
        since=datetime(2026, 8, 1, tzinfo=UTC),
        until=datetime(2026, 9, 1, tzinfo=UTC),
        spend_usd=Decimal(spend),
        requests=kwargs.pop("requests", 1000),  # type: ignore[arg-type]
        unpriced_events=kwargs.pop("unpriced_events", 0),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


# --- scope ------------------------------------------------------------------------


def test_a_misspelled_scope_field_is_rejected() -> None:
    # A budget scoped to a typo would watch the entire estate and never fire, which is
    # the most dangerous possible failure for a spend limit.
    with pytest.raises(InvalidScopeError):
        make_budget(scope={"porject": "checkout"})


def test_scope_becomes_bound_parameters() -> None:
    since = datetime(2026, 8, 1, tzinfo=UTC)
    until = datetime(2026, 9, 1, tzinfo=UTC)
    filters = budgets.scope_filters(
        {"project": ["a", "b"], "feature": "search", "tags": {"team": "ml"}}, since, until
    )
    assert filters.project == ("a", "b")
    assert filters.feature == ("search",)
    assert filters.tags == {"team": "ml"}

    _, params = filters.where()
    assert params["project"] == ["a", "b"]


def test_tags_scope_must_be_an_object() -> None:
    with pytest.raises(InvalidScopeError):
        make_budget(scope={"tags": ["team"]})


# --- periods ----------------------------------------------------------------------


def test_monthly_period_is_the_calendar_month() -> None:
    budget = make_budget()
    start, since, until = budgets.period_bounds(budget, datetime(2026, 8, 21, 13, 4, tzinfo=UTC))
    assert start == datetime(2026, 8, 1, tzinfo=UTC).date()
    assert since == datetime(2026, 8, 1, tzinfo=UTC)
    assert until == datetime(2026, 9, 1, tzinfo=UTC)


def test_monthly_period_rolls_over_the_year() -> None:
    budget = make_budget()
    _, since, until = budgets.period_bounds(budget, datetime(2026, 12, 31, 23, 59, tzinfo=UTC))
    assert (since.year, since.month) == (2026, 12)
    assert (until.year, until.month) == (2027, 1)


def test_rolling_period_looks_back_n_days_and_re_anchors_daily() -> None:
    budget = make_budget(period=Period.ROLLING, rolling_days=7)
    now = datetime(2026, 8, 21, 13, 0, tzinfo=UTC)
    start, since, until = budgets.period_bounds(budget, now)

    assert until == now
    assert (until - since).days == 7
    # The window slides continuously, so the fire-once key is the as-of day: an
    # over-budget rolling budget alerts at most once a day, and can alert again
    # tomorrow if it is still over.
    assert start == now.date()
    tomorrow_start, _, _ = budgets.period_bounds(budget, now.replace(day=22))
    assert tomorrow_start != start


def test_a_rolling_budget_without_a_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="rolling_days"):
        make_budget(period=Period.ROLLING, rolling_days=None)


def test_a_naive_now_is_treated_as_utc() -> None:
    budget = make_budget()
    _, since, _ = budgets.period_bounds(budget, datetime(2026, 8, 21, 13, 0))
    assert since.tzinfo is UTC


# --- thresholds -------------------------------------------------------------------


def test_crossed_returns_every_reached_threshold_lowest_first() -> None:
    status = make_status("85")
    assert status.crossed() == (Decimal("0.5"), Decimal("0.8"))
    assert status.utilization == Decimal("0.85")
    assert status.remaining_usd == Decimal("15")


def test_nothing_is_crossed_below_the_first_threshold() -> None:
    assert make_status("10").crossed() == ()


def test_unpriced_events_are_reported_as_a_blind_spot() -> None:
    # sum(cost_usd) skips NULLs, so an unresolved model makes a budget read as *under*
    # budget. The count travels with the status so the gap is never invisible.
    clean = make_status("50")
    blind = make_status("50", unpriced_events=42)
    assert not clean.has_blind_spot
    assert blind.has_blind_spot


def test_forecast_drives_projected_utilization() -> None:
    status = make_status("40", forecast_usd=Decimal("130"))
    assert status.projected_utilization == Decimal("1.3")
    assert status.crossed() == ()  # the forecast warns; it does not fire the alert


# --- webhook signing ---------------------------------------------------------------


def test_signature_round_trips() -> None:
    body = b'{"type":"budget.threshold_crossed"}'
    header = budgets.sign("s3cret", body, int(time.time()))
    assert budgets.verify("s3cret", body, header)


def test_a_tampered_body_fails_verification() -> None:
    body = b'{"spend_usd":"50"}'
    header = budgets.sign("s3cret", body, int(time.time()))
    assert not budgets.verify("s3cret", b'{"spend_usd":"5"}', header)


def test_a_replayed_signature_expires() -> None:
    body = b"{}"
    stale = budgets.sign("s3cret", body, int(time.time()) - 3600)
    assert not budgets.verify("s3cret", body, stale, tolerance_seconds=300)


def test_a_wrong_secret_fails_verification() -> None:
    body = b"{}"
    header = budgets.sign("s3cret", body, int(time.time()))
    assert not budgets.verify("other", body, header)


def test_a_malformed_signature_header_is_rejected_not_raised() -> None:
    assert not budgets.verify("s3cret", b"{}", "garbage")
    assert not budgets.verify("s3cret", b"{}", "t=notanumber,v1=abc")


# --- delivery ----------------------------------------------------------------------


def make_alert(budget: Budget) -> Alert:
    return Alert(
        id=1,
        budget=budget,
        period_start=datetime(2026, 8, 1, tzinfo=UTC).date(),
        threshold=Decimal("0.8"),
        spend_usd=Decimal("85"),
        amount_usd=Decimal("100"),
        fired_at=datetime(2026, 8, 21, tzinfo=UTC),
    )


def client_returning(*statuses: int) -> tuple[httpx.Client, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    codes = iter(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(next(codes, statuses[-1]))

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def test_a_signed_payload_is_delivered() -> None:
    budget = make_budget(webhook_url="https://example.test/hook", webhook_secret="s3cret")
    alert = make_alert(budget)
    client, seen = client_returning(204)

    ok, error = budgets.deliver(alert, budgets.alert_payload(alert), client=client)

    assert (ok, error) == (True, None)
    request = seen[0]
    assert budgets.verify("s3cret", request.content, request.headers[budgets.SIGNATURE_HEADER])


def test_server_errors_are_retried_then_reported() -> None:
    budget = make_budget(webhook_url="https://example.test/hook")
    client, seen = client_returning(500, 502, 503)
    slept: list[float] = []

    ok, error = budgets.deliver(make_alert(budget), {}, client=client, sleep=slept.append)

    assert ok is False
    assert error == "HTTP 503"
    assert len(seen) == budgets.WEBHOOK_ATTEMPTS
    assert slept == [0.5, 1.0]  # exponential, and no sleep after the final attempt


def test_a_client_error_is_not_retried() -> None:
    # 404 means the endpoint is wrong. Retrying just repeats the mistake.
    budget = make_budget(webhook_url="https://example.test/hook")
    client, seen = client_returning(404)

    ok, error = budgets.deliver(make_alert(budget), {}, client=client, sleep=lambda _: None)

    assert (ok, error) == (False, "HTTP 404")
    assert len(seen) == 1


def test_a_transient_failure_recovers_on_retry() -> None:
    budget = make_budget(webhook_url="https://example.test/hook")
    client, seen = client_returning(503, 200)

    ok, _ = budgets.deliver(make_alert(budget), {}, client=client, sleep=lambda _: None)

    assert ok is True
    assert len(seen) == 2


def test_network_errors_are_caught_not_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    budget = make_budget(webhook_url="https://example.test/hook")
    client = httpx.Client(transport=httpx.MockTransport(handler))

    ok, error = budgets.deliver(make_alert(budget), {}, client=client, sleep=lambda _: None)

    assert ok is False
    assert error is not None
    assert "ConnectError" in error


def test_an_unconfigured_webhook_is_not_an_error_path() -> None:
    ok, error = budgets.deliver(make_alert(make_budget()), {})
    assert (ok, error) == (False, "no webhook configured")


def test_the_payload_carries_the_blind_spot() -> None:
    budget = make_budget(webhook_url="https://example.test/hook")
    alert = make_alert(budget)
    payload = budgets.alert_payload(alert, make_status("85", budget=budget, unpriced_events=7))

    assert payload["unpriced_events"] == 7
    assert payload["threshold"] == 0.8
    assert payload["spend_usd"] == "85"
    assert payload["budget"]["scope"] == {"project": "checkout"}


def test_alerts_cannot_fire_for_an_unsaved_budget() -> None:
    # Without an id there is no UNIQUE row to de-duplicate against, so the fire-once
    # guarantee would silently not hold.
    with pytest.raises(ValueError, match="persisted"):
        budgets.fire_alerts(None, make_status("85", budget=make_budget(id=None)))  # type: ignore[arg-type]
