"""Storage against a real Postgres: partitioning, idempotency, attribution, rollups."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from tokenomics.finops import tokens as token_service
from tokenomics.models import TokenVector, UsageEvent
from tokenomics.pricing import cost
from tokenomics.storage import queries, repository
from tokenomics.storage.queries import InvalidDimensionError

pytestmark = pytest.mark.integration


def test_events_insert_and_aggregate(conn, make_event, window) -> None:
    events = [make_event(n=i) for i in range(5)]
    assert repository.insert_events(conn, events) == 5

    rows = queries.spend_breakdown(conn, filters=window(), group_by=["project"])

    assert rows[0]["project"] == "checkout"
    assert rows[0]["requests"] == 5
    assert rows[0]["cost_usd"] == sum(e.cost.total_usd for e in events if e.cost)


def test_re_inserting_the_same_spans_does_not_double_bill(conn, make_event, now) -> None:
    # Collector retries and importer re-runs are normal operation, not an error case.
    events = [make_event(n=i) for i in range(3)]
    assert repository.insert_events(conn, events) == 3
    assert repository.insert_events(conn, events) == 0

    total = repository.total_spend(conn, now - timedelta(days=1), now + timedelta(days=1))
    assert total == sum(e.cost.total_usd for e in events if e.cost)


def test_an_unpriced_event_is_stored_as_null_not_zero(conn, window, now) -> None:
    from tokenomics.pricing.engine import default_engine

    engine = default_engine()
    event = engine.price(
        UsageEvent(
            trace_id="trace-unpriced",
            span_id="span-unpriced",
            ts=now,
            request_model="acme/never-heard-of-it",
            tokens=TokenVector(input=1_000, output=100),
        )
    ).event
    assert event.cost is None
    repository.insert_events(conn, [event])

    with conn.cursor() as cur:
        cur.execute("SELECT cost_usd FROM usage_event WHERE trace_id = 'trace-unpriced'")
        row = cur.fetchone()
    assert row is not None
    assert row[0] is None  # NULL means "we could not price it", never "it was free"

    rows = queries.spend_breakdown(conn, filters=window(), group_by=[])
    assert rows[0]["unpriced_events"] == 1

    unpriced = queries.unpriced_summary(conn, filters=window())
    assert unpriced[0]["model"] == "acme/never-heard-of-it"


def test_attribution_filters_and_grouping(conn, make_event, window) -> None:
    repository.insert_events(
        conn,
        [
            make_event(n=10, project="checkout", feature="search"),
            make_event(n=11, project="checkout", feature="summarize"),
            make_event(n=12, project="support", feature="search"),
        ],
    )

    rows = queries.spend_breakdown(
        conn, filters=window(project=("checkout",)), group_by=["feature"]
    )
    assert {r["feature"] for r in rows} == {"search", "summarize"}

    combined = queries.spend_breakdown(conn, filters=window(), group_by=["project", "feature"])
    assert len(combined) == 3


def test_arbitrary_tags_are_queryable(conn, make_event, window) -> None:
    repository.insert_events(
        conn,
        [
            make_event(n=20, tags={"team": "ml", "tier": "free"}),
            make_event(n=21, tags={"team": "platform"}),
        ],
    )

    rows = queries.spend_breakdown(conn, filters=window(tags={"team": "ml"}), group_by=[])
    assert rows[0]["requests"] == 1


def test_a_group_by_dimension_outside_the_whitelist_is_rejected(conn, window) -> None:
    # The whitelist is the injection boundary: dimensions become SQL identifiers.
    with pytest.raises(InvalidDimensionError):
        queries.spend_breakdown(
            conn, filters=window(), group_by=["project; DROP TABLE usage_event"]
        )

    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('usage_event')")
        row = cur.fetchone()
    assert row is not None and row[0] is not None


def test_events_land_in_the_right_monthly_partition(conn, make_event, now) -> None:
    old = now - timedelta(days=45)
    repository.insert_events(conn, [make_event(n=30, ts=old), make_event(n=31, ts=now)])

    with conn.cursor() as cur:
        cur.execute(
            "SELECT tableoid::regclass::text, count(*) FROM usage_event "
            "WHERE trace_id IN ('trace-0030', 'trace-0031') GROUP BY 1"
        )
        partitions = dict(cur.fetchall())

    assert len(partitions) == 2  # different months, different physical tables
    assert set(partitions.values()) == {1}


def test_rollups_recompute_rather_than_accumulate(conn, make_event, now) -> None:
    since, until = now - timedelta(hours=1), now + timedelta(hours=1)
    repository.insert_events(conn, [make_event(n=40)])
    repository.refresh_rollups(conn, since, until)
    repository.refresh_rollups(conn, since, until)  # late-arriving spans re-run this

    with conn.cursor() as cur:
        cur.execute(
            "SELECT sum(requests), sum(cost_usd) FROM usage_rollup_hourly WHERE bucket >= %s",
            (since,),
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == 1  # not 2: the refresh is idempotent


def test_unit_economics_over_real_rows(conn, make_event, window) -> None:
    repository.insert_events(
        conn,
        [
            make_event(n=50, subject="u-1"),
            make_event(n=51, subject="u-2"),
            make_event(
                n=52,
                subject="u-2",
                tokens=TokenVector(input=10_000, output=1_000, cache_read=8_000),
            ),
        ],
    )

    rows = queries.unit_economics(conn, filters=window(), group_by=["project"])
    row = rows[0]

    assert row["requests"] == 3
    assert row["subjects"] == 2
    assert row["cost_per_request"] == Decimal(row["cost_usd"]) / 3
    assert row["cost_per_subject"] == Decimal(row["cost_usd"]) / 2
    assert 0 < row["cache_hit_rate"] < 1


# --------------------------------------------------------------- token economics


def test_aggregates_carry_reasoning_and_per_component_cost(conn, make_event, window) -> None:
    """These columns were written from day one but read by nothing, which is the same
    as not having them."""
    repository.insert_events(
        conn,
        [
            make_event(
                n=60,
                model="o3",
                tokens=TokenVector(input=1_000, output=2_000, reasoning=900),
            )
        ],
    )

    row = queries.spend_breakdown(conn, filters=window(), group_by=["project"])[0]

    assert row["reasoning_tokens"] == 900
    assert row["billable_input_tokens"] == 1_000
    components = ("input_usd", "output_usd", "cache_read_usd", "cache_write_usd", "reasoning_usd")
    assert sum(Decimal(row[key]) for key in components) == Decimal(row["cost_usd"])


def test_billable_input_excludes_what_came_from_cache(conn, make_event, window) -> None:
    repository.insert_events(
        conn,
        [
            make_event(
                n=61,
                model="claude-sonnet-4-5",
                tokens=TokenVector(input=10_000, output=500, cache_read=7_000, cache_write=1_000),
            )
        ],
    )

    row = queries.spend_breakdown(conn, filters=window(), group_by=["project"])[0]

    assert row["input_tokens"] == 10_000  # the reported total is unchanged
    assert row["billable_input_tokens"] == 2_000  # what was billed at the input rate


@pytest.mark.parametrize(
    "tokens",
    [
        TokenVector(input=10_000, output=1_000, cache_read=8_000, cache_write=1_000),
        TokenVector(input=1_000, output=500),
        # Inconsistent instrumentation: more cached tokens than input tokens.
        TokenVector(input=100, output=10, cache_read=90, cache_write=50),
        TokenVector(input=0, output=0),
        TokenVector(input=50, output=5, cache_read=50, cache_write=50),
    ],
)
def test_sql_partition_agrees_with_the_cost_function(conn, make_event, window, tokens) -> None:
    """The SQL restates ``pricing.cost._partition_input`` in another language.

    That duplication is deliberate -- aggregating in Postgres is the whole point -- but
    it can only be trusted if something checks the two against each other, especially
    on the inconsistent inputs where the naive expression and the real one diverge.
    """
    repository.insert_events(conn, [make_event(n=62, model="claude-sonnet-4-5", tokens=tokens)])

    row = queries.token_economics(conn, filters=window(), group_by=["project"])[0]
    expected_input, expected_read, expected_write, _ = cost._partition_input(tokens)

    assert row["billable_input_tokens"] == expected_input
    assert row["billable_cache_read_tokens"] == expected_read
    assert row["billable_cache_write_tokens"] == expected_write
    # Whatever the input, the buckets must still partition the reported total.
    assert expected_input + expected_read + expected_write == tokens.input


def test_token_components_partition_the_reported_total(conn, make_event, window) -> None:
    repository.insert_events(
        conn,
        [
            make_event(
                n=63,
                model="claude-sonnet-4-5",
                tokens=TokenVector(input=9_000, output=800, cache_read=6_000, cache_write=500),
            ),
            make_event(n=64, model="o3", tokens=TokenVector(input=500, output=900, reasoning=400)),
            # Inconsistent instrumentation must not break the sum.
            make_event(
                n=65,
                model="claude-sonnet-4-5",
                tokens=TokenVector(input=100, output=10, cache_read=90, cache_write=50),
            ),
        ],
    )

    row = queries.token_economics(conn, filters=window(), group_by=["project"])[0]
    derived = token_service.derive(row)

    assert sum(derived["token_components"].values()) == derived["total_tokens"]
    assert derived["total_tokens"] == row["input_tokens"] + row["output_tokens"]
    assert sum(derived["cost_components"].values()) == Decimal(row["cost_usd"])


def test_cache_savings_are_priced_from_the_rates_each_event_was_billed_at(
    conn, make_event, window
) -> None:
    repository.insert_events(
        conn,
        [
            make_event(
                n=66,
                model="claude-sonnet-4-5",
                tokens=TokenVector(input=10_000, output=100, cache_read=9_000),
            )
        ],
    )

    derived = token_service.derive(
        queries.token_economics(conn, filters=window(), group_by=["project"])[0]
    )

    # A cache read is ~0.1x input, so reading 9k tokens must be a clear net win.
    assert derived["cache_savings_usd"] > 0
    assert derived["cache_write_premium_usd"] == 0
    assert derived["net_cache_benefit_usd"] == derived["cache_savings_usd"]
    assert derived["cache_priced_coverage"] == pytest.approx(1.0)


def test_unpriced_cached_traffic_lowers_the_coverage_it_reports(conn, window) -> None:
    """A model we cannot price contributes cached tokens but no rate to value them at.

    Silently leaving it out would understate the saving with no sign that it had; the
    coverage figure is how the gap stays visible.
    """
    repository.insert_events(
        conn,
        [
            UsageEvent(
                trace_id="trace-0067",
                span_id="span-0067",
                ts=window().since + timedelta(hours=1),
                provider="mystery",
                request_model="model-that-does-not-exist",
                tokens=TokenVector(input=1_000, output=100, cache_read=800),
                cost=None,
            )
        ],
    )

    derived = token_service.derive(
        queries.token_economics(conn, filters=window(), group_by=["project"])[0]
    )

    assert derived["cache_priced_coverage"] == 0.0
    assert derived["cache_savings_usd"] == 0


def test_rollup_carries_the_token_and_component_columns(conn, make_event, now) -> None:
    since, until = now - timedelta(hours=1), now + timedelta(hours=1)
    event = make_event(n=68, model="o3", tokens=TokenVector(input=1_000, output=800, reasoning=300))
    repository.insert_events(conn, [event])
    repository.refresh_rollups(conn, since, until)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT sum(reasoning_tokens), sum(billable_input_tokens), "
            "sum(input_usd) + sum(output_usd) + sum(cache_read_usd) "
            "+ sum(cache_write_usd) + sum(reasoning_usd), sum(cost_usd) "
            "FROM usage_rollup_hourly WHERE bucket >= %s",
            (since,),
        )
        row = cur.fetchone()

    assert row is not None
    assert row[0] == 300
    assert row[1] == 1_000
    assert row[2] == row[3]  # components add up to the total in the rollup too
