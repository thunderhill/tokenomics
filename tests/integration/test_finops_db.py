"""FinOps against a real Postgres: the fire-once state machine and DB-backed replay."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import httpx
import pytest

from tokenomics.finops import budgets, reports, whatif
from tokenomics.finops.budgets import Budget, Period
from tokenomics.finops.reports import UNALLOCATED, Allocation
from tokenomics.models import TokenVector

pytestmark = pytest.mark.integration


# --- budgets ----------------------------------------------------------------------


def test_a_budget_round_trips_through_the_database(conn) -> None:
    saved = budgets.create_budget(
        conn,
        Budget(
            name="checkout monthly",
            amount_usd=Decimal("500"),
            scope={"project": "checkout", "tags": {"team": "ml"}},
            thresholds=(Decimal("0.75"), Decimal("1.0")),
            webhook_url="https://example.test/hook",
        ),
    )

    assert saved.id is not None
    loaded = budgets.get_budget(conn, saved.id)
    assert loaded is not None
    assert loaded.amount_usd == Decimal("500")
    assert loaded.scope == {"project": "checkout", "tags": {"team": "ml"}}
    assert loaded.thresholds == (Decimal("0.75"), Decimal("1.0"))
    assert [b.name for b in budgets.list_budgets(conn)] == ["checkout monthly"]


def test_a_budget_only_sees_spend_inside_its_scope(conn, seed, now) -> None:
    mine = seed(n=100, project="checkout")
    seed(n=101, project="support")

    budget = budgets.create_budget(
        conn, Budget(name="checkout", amount_usd=Decimal("1"), scope={"project": "checkout"})
    )
    status = budgets.evaluate(conn, budget, now=now)

    assert status.spend_usd == mine
    assert status.requests == 1


def test_each_threshold_fires_exactly_once_per_period(conn, seed, now) -> None:
    spend = seed(n=110)
    budget = budgets.create_budget(
        conn,
        Budget(
            name="tight",
            amount_usd=spend,  # exactly 100% utilization
            scope={"project": "checkout"},
        ),
    )

    status = budgets.evaluate(conn, budget, now=now)
    first = budgets.fire_alerts(conn, status)
    second = budgets.fire_alerts(conn, budgets.evaluate(conn, budget, now=now))

    assert [a.threshold for a in first] == [Decimal("0.5"), Decimal("0.8"), Decimal("1.0")]
    # Budgets are evaluated on a schedule. Without the fire-once constraint this is the
    # loop that trains people to mute the alerts.
    assert second == []


def test_only_the_newly_crossed_threshold_fires(conn, seed, now) -> None:
    spend = seed(n=120)
    budget = budgets.create_budget(
        conn, Budget(name="growing", amount_usd=spend * 2, scope={"project": "checkout"})
    )

    at_fifty = budgets.fire_alerts(conn, budgets.evaluate(conn, budget, now=now))
    seed(n=121)  # spend rises to 100%
    at_hundred = budgets.fire_alerts(conn, budgets.evaluate(conn, budget, now=now))

    assert [a.threshold for a in at_fifty] == [Decimal("0.5")]
    assert [a.threshold for a in at_hundred] == [Decimal("0.8"), Decimal("1.0")]


def test_a_new_period_can_fire_again(conn, seed, now) -> None:
    spend = seed(n=130)
    budget = budgets.create_budget(
        conn, Budget(name="monthly", amount_usd=spend, scope={"project": "checkout"})
    )

    budgets.fire_alerts(conn, budgets.evaluate(conn, budget, now=now))

    # The same spend, attributed to the next calendar month: a budget is a *per-period*
    # limit, so the state machine re-arms rather than staying silent forever.
    this_month = budgets.evaluate(conn, budget, now=now)
    next_month = replace(this_month, period_start=(now.replace(day=1) + timedelta(days=32)).date())
    rearmed = budgets.fire_alerts(conn, next_month)
    assert [a.threshold for a in rearmed] == [Decimal("0.5"), Decimal("0.8"), Decimal("1.0")]


def test_a_rolling_budget_measures_a_sliding_window(conn, seed, window, now) -> None:
    seed(n=140, ts=now - timedelta(days=20))
    recent = seed(n=141, ts=now - timedelta(days=1))

    budget = budgets.create_budget(
        conn,
        Budget(
            name="rolling 7d",
            amount_usd=Decimal("100"),
            scope={"project": "checkout"},
            period=Period.ROLLING,
            rolling_days=7,
        ),
    )
    status = budgets.evaluate(conn, budget, now=now)

    assert status.spend_usd == recent  # the 20-day-old event is outside the window


def test_the_cycle_delivers_a_signed_webhook_and_records_it(conn, seed, now) -> None:
    spend = seed(n=150)
    budget = budgets.create_budget(
        conn,
        Budget(
            name="webhooked",
            amount_usd=spend,
            scope={"project": "checkout"},
            thresholds=(Decimal("1.0"),),
            webhook_url="https://example.test/hook",
            webhook_secret="s3cret",
        ),
    )

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fired = budgets.run_budget_cycle(conn, now=now, client=client, budgets=[budget])

    assert len(fired) == 1
    assert len(seen) == 1
    assert budgets.verify("s3cret", seen[0].content, seen[0].headers[budgets.SIGNATURE_HEADER])

    recorded = budgets.recent_alerts(conn, budget_id=budget.id)
    assert recorded[0]["delivered"] is True
    assert recorded[0]["delivery_error"] is None


def test_a_failed_delivery_is_recorded_without_losing_the_alert(conn, seed, now) -> None:
    spend = seed(n=160)
    budget = budgets.create_budget(
        conn,
        Budget(
            name="broken hook",
            amount_usd=spend,
            scope={"project": "checkout"},
            thresholds=(Decimal("1.0"),),
            webhook_url="https://example.test/hook",
        ),
    )
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(404)))

    fired = budgets.run_budget_cycle(conn, now=now, client=client, budgets=[budget])

    assert len(fired) == 1  # the threshold still counts as crossed
    recorded = budgets.recent_alerts(conn, budget_id=budget.id)
    assert recorded[0]["delivered"] is False
    assert recorded[0]["delivery_error"] == "HTTP 404"


def test_deleting_a_budget_takes_its_alerts_with_it(conn, seed, now) -> None:
    spend = seed(n=170)
    budget = budgets.create_budget(
        conn, Budget(name="doomed", amount_usd=spend, scope={"project": "checkout"})
    )
    budgets.fire_alerts(conn, budgets.evaluate(conn, budget, now=now))
    assert budget.id is not None

    assert budgets.delete_budget(conn, budget.id) is True
    assert budgets.recent_alerts(conn, budget_id=budget.id) == []


# --- what-if ----------------------------------------------------------------------


def test_load_samples_collapses_identical_shapes(conn, seed, window) -> None:
    for i in range(5):
        seed(n=200 + i)  # same token vector every time
    seed(n=210, tokens=TokenVector(input=50_000, output=2_000))

    samples, scale = whatif.load_samples(conn, filters=window())

    assert scale == 1.0
    assert len(samples) == 2  # two distinct shapes, not six rows
    assert sum(s.requests for s in samples) == 6
    assert all(s.baseline_usd is not None for s in samples)


def test_load_samples_reprices_real_traffic(conn, engine, seed, window) -> None:
    for i in range(3):
        seed(n=220 + i, model="gpt-4o")

    samples, scale = whatif.load_samples(conn, filters=window())
    result = whatif.simulate(engine, samples, target_model="gpt-4o-mini", scale_factor=scale)

    assert result.requests == 3
    assert result.baseline_usd > 0
    assert result.projected_usd < result.baseline_usd
    assert result.is_complete


def test_load_samples_scales_up_when_traffic_is_too_varied(conn, engine, seed, window) -> None:
    for i in range(6):
        seed(n=230 + i, tokens=TokenVector(input=1_000 + i, output=100))

    samples, scale = whatif.load_samples(conn, filters=window(), max_shapes=3)

    assert len(samples) == 3
    assert scale == pytest.approx(2.0)
    result = whatif.simulate(engine, samples, target_model="gpt-4o", scale_factor=scale)
    assert whatif.WARN_SAMPLED in result.warnings
    assert not result.is_complete


def test_an_unpriced_event_makes_its_shape_an_unpriced_baseline(conn, seed, window) -> None:
    seed(n=240, model="acme/never-heard-of-it")

    samples, _ = whatif.load_samples(conn, filters=window())

    assert [s.baseline_usd for s in samples] == [None]


# --- reports ----------------------------------------------------------------------


def test_chargeback_over_real_traffic(conn, seed, window) -> None:
    seed(n=300, project="checkout")
    seed(n=301, project="checkout")
    seed(n=302, project="support")
    seed(n=303, project="unknown")

    report = reports.chargeback(conn, filters=window(), group_by=("project",))

    names = [line.name for line in report.lines]
    assert names[:2] == ["checkout", "support"]  # ordered by spend
    assert UNALLOCATED in names
    assert report.unallocated_usd > 0
    assert report.total_usd == sum(line.total_usd for line in report.lines)


def test_spread_chargeback_bills_the_whole_estate(conn, seed, window) -> None:
    seed(n=310, project="checkout")
    seed(n=311, project="support")
    seed(n=312, project="unknown")

    shown = reports.chargeback(conn, filters=window(), group_by=("project",))
    charged = reports.chargeback(
        conn, filters=window(), group_by=("project",), allocation=Allocation.SPREAD
    )

    assert UNALLOCATED not in {line.name for line in charged.lines}
    # Same money, different owners -- to the cent.
    assert charged.total_usd.quantize(Decimal("0.01")) == shown.total_usd.quantize(Decimal("0.01"))


def test_showback_by_feature_and_csv_export(conn, seed, window) -> None:
    seed(n=320, feature="search")
    seed(n=321, feature="summarize")

    report = reports.showback(conn, filters=window(), group_by=("project", "feature"))
    csv_text = reports.to_csv(report)

    assert report.kind == "showback"
    assert csv_text.splitlines()[0].startswith("project,feature,")
    assert len(csv_text.splitlines()) == 3
