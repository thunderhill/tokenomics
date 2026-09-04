"""Chargeback allocation, unit economics and export."""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from tokenomics.finops.reports import (
    UNALLOCATED,
    Allocation,
    ReportLine,
    build_report,
    spread,
    to_csv,
)

SINCE = datetime(2026, 8, 1, tzinfo=UTC)
UNTIL = datetime(2026, 9, 1, tzinfo=UTC)


def row(project: str | None, cost: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "project": project,
        "requests": kwargs.get("requests", 100),
        "subjects": kwargs.get("subjects", 10),
        "cost_usd": Decimal(cost),
        "input_tokens": kwargs.get("input_tokens", 100_000),
        "output_tokens": kwargs.get("output_tokens", 10_000),
        "cache_read_tokens": kwargs.get("cache_read_tokens", 0),
        "cache_write_tokens": kwargs.get("cache_write_tokens", 0),
        "unpriced_events": kwargs.get("unpriced_events", 0),
    }


def report(rows: list[dict[str, Any]], **kwargs: Any) -> Any:
    return build_report(rows, group_by=("project",), since=SINCE, until=UNTIL, **kwargs)


def test_lines_are_ordered_by_spend() -> None:
    result = report([row("a", "10"), row("b", "50"), row("c", "30")])
    assert [line.name for line in result.lines] == ["b", "c", "a"]
    assert result.total_usd == Decimal("90")


def test_null_and_unknown_ownership_collapse_into_one_line() -> None:
    # They are the same fact -- nobody claimed this spend -- and splitting them across
    # two lines makes the unallocated total look smaller than it is.
    result = report([row("checkout", "60"), row(None, "10"), row("unknown", "30")])

    unallocated = [line for line in result.lines if line.name == UNALLOCATED]
    assert len(unallocated) == 1
    assert unallocated[0].direct_usd == Decimal("40")
    assert unallocated[0].requests == 200
    assert result.unallocated_usd == Decimal("40")


def test_showback_keeps_unallocated_visible() -> None:
    result = report([row("checkout", "60"), row(None, "40")], allocation=Allocation.SHOW)
    assert result.total_usd == Decimal("100")
    assert {line.name for line in result.lines} == {"checkout", UNALLOCATED}


def test_spread_allocates_pro_rata_and_removes_the_unallocated_line() -> None:
    result = report([row("a", "75"), row("b", "25"), row(None, "20")], allocation=Allocation.SPREAD)

    by_name = {line.name: line for line in result.lines}
    assert UNALLOCATED not in by_name
    assert by_name["a"].allocated_usd == Decimal("15.00")
    assert by_name["b"].allocated_usd == Decimal("5.00")
    assert by_name["a"].total_usd == Decimal("90.00")
    assert result.total_usd == Decimal("120.00")


def test_spread_is_penny_exact() -> None:
    # $10 over three equal owners is $3.333...; naive rounding loses a cent and a
    # chargeback report that does not add up gets rejected, rightly.
    lines = [
        ReportLine(
            key=(name,),
            requests=1,
            subjects=1,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            direct_usd=Decimal("1"),
            allocated_usd=Decimal(0),
            unpriced_events=0,
        )
        for name in ("a", "b", "c")
    ]

    allocated = spread(lines, Decimal("10"))

    assert sum(line.allocated_usd for line in allocated) == Decimal("10.00")


def test_spread_falls_back_to_request_share_when_nothing_is_priced() -> None:
    result = report(
        [row("a", "0", requests=300), row("b", "0", requests=100), row(None, "8")],
        allocation=Allocation.SPREAD,
    )
    by_name = {line.name: line for line in result.lines}
    assert by_name["a"].allocated_usd == Decimal("6.00")
    assert by_name["b"].allocated_usd == Decimal("2.00")


def test_spreading_onto_nobody_does_not_delete_the_money() -> None:
    result = report([row(None, "40")], allocation=Allocation.SPREAD)

    assert result.total_usd == Decimal("40")
    assert result.lines[0].name == UNALLOCATED
    # The report records the policy that actually applied, not the one requested.
    assert result.allocation is Allocation.SHOW


def test_unit_economics_come_from_the_allocated_total() -> None:
    result = report(
        [row("a", "90", requests=300, subjects=30), row(None, "30")],
        allocation=Allocation.SPREAD,
    )
    line = result.lines[0]

    assert line.total_usd == Decimal("120.00")
    assert line.cost_per_request == Decimal("120.00") / 300
    assert line.cost_per_subject == Decimal("120.00") / 30
    assert line.cost_per_1k_tokens == Decimal("120.00") / 110_000 * 1000


def test_cache_hit_rate_is_a_share_of_input() -> None:
    result = report([row("a", "10", input_tokens=100_000, cache_read_tokens=80_000)])
    assert result.lines[0].cache_hit_rate == 0.8


def test_lines_without_traffic_report_none_rather_than_zero() -> None:
    result = report([row("a", "10", requests=0, subjects=0, input_tokens=0, output_tokens=0)])
    line = result.lines[0]
    assert line.cost_per_request is None
    assert line.cost_per_subject is None
    assert line.cost_per_1k_tokens is None
    assert line.cache_hit_rate is None


def test_unpriced_events_survive_into_the_report() -> None:
    result = report([row("a", "10", unpriced_events=5), row("b", "10")])
    assert result.unpriced_events == 5


def test_shares_sum_to_one() -> None:
    result = report([row("a", "60"), row("b", "30"), row("c", "10")])
    assert sum(result.share(line) for line in result.lines) == 1.0


def test_multi_dimension_grouping_keys_every_line() -> None:
    rows = [
        {**row("checkout", "10"), "feature": "search"},
        {**row("checkout", "20"), "feature": "summarize"},
    ]
    result = build_report(rows, group_by=("project", "feature"), since=SINCE, until=UNTIL)

    assert [line.name for line in result.lines] == ["checkout / summarize", "checkout / search"]
    assert result.to_dict()["lines"][0]["feature"] == "summarize"


def test_csv_export_is_flat_and_complete() -> None:
    result = report([row("checkout", "60"), row(None, "40")])

    parsed = list(csv.DictReader(io.StringIO(to_csv(result))))

    assert parsed[0]["project"] == "checkout"
    assert parsed[0]["total_usd"] == "60"
    assert parsed[1]["project"] == UNALLOCATED
    assert {r["project"] for r in parsed} == {"checkout", UNALLOCATED}


def test_csv_writes_empty_cells_for_undefined_ratios() -> None:
    result = report([row("a", "10", requests=0, subjects=0)])
    parsed = list(csv.DictReader(io.StringIO(to_csv(result))))
    assert parsed[0]["cost_per_request"] == ""


def test_to_dict_is_json_safe_and_keeps_money_exact() -> None:
    import json

    result = report([row("a", "0.123456789012")])
    payload = json.loads(json.dumps(result.to_dict()))

    assert payload["lines"][0]["total_usd"] == "0.123456789012"
    assert payload["kind"] == "chargeback"
    assert payload["currency"] == "USD"


def test_an_empty_period_reports_zero_not_an_error() -> None:
    result = report([])
    assert result.lines == ()
    assert result.total_usd == Decimal(0)
    assert result.unpriced_events == 0


# ------------------------------------------------------- token and component columns


def token_row(project: str, cost: str, **kwargs: Any) -> dict[str, Any]:
    """A row shaped the way the widened `_METRICS` block returns it."""
    return {
        **row(project, cost, **kwargs),
        "reasoning_tokens": kwargs.get("reasoning_tokens", 4_000),
        "billable_input_tokens": kwargs.get("billable_input_tokens", 80_000),
        "input_usd": Decimal("0.24"),
        "output_usd": Decimal("0.60"),
        "cache_read_usd": Decimal("0.006"),
        "cache_write_usd": Decimal("0.075"),
        "reasoning_usd": Decimal("0.08"),
    }


def test_lines_carry_reasoning_and_the_component_split() -> None:
    result = report([token_row("checkout", "1.001")])
    line = result.lines[0]

    assert line.reasoning_tokens == 4_000
    assert line.billable_input_tokens == 80_000
    assert line.output_usd == Decimal("0.60")
    assert line.reasoning_share == 0.4


def test_output_cost_share_is_measured_against_direct_spend() -> None:
    """Allocated spend is somebody else's tokens, so it cannot dilute this ratio."""
    result = report(
        [token_row("checkout", "1.001"), token_row(None, "0.5")],
        allocation=Allocation.SPREAD,
    )
    line = result.lines[0]

    assert line.allocated_usd > 0
    assert line.output_cost_share == float(line.output_usd / line.direct_usd)


def test_spreading_moves_money_without_moving_tokens() -> None:
    result = report(
        [token_row("checkout", "3"), token_row("support", "1"), token_row(None, "0.4")],
        allocation=Allocation.SPREAD,
    )

    for line in result.lines:
        # The unowned line's tokens stay unowned; only its cost is redistributed.
        assert line.reasoning_tokens == 4_000
        assert line.billable_input_tokens == 80_000
        assert line.input_usd == Decimal("0.24")


def test_unowned_rows_merge_their_token_columns_too() -> None:
    """NULL and the literal 'unknown' are the same fact and land on one line."""
    result = report([token_row(None, "1"), token_row("unknown", "1")])
    line = next(item for item in result.lines if item.key[0] == UNALLOCATED)

    assert line.reasoning_tokens == 8_000
    assert line.billable_input_tokens == 160_000
    assert line.output_usd == Decimal("1.20")


def test_blended_rate_uses_the_inclusive_total_not_the_component_sum() -> None:
    line = report([token_row("checkout", "1.1")]).lines[0]

    assert line.total_tokens == 110_000
    assert line.usd_per_1m_tokens == Decimal("1.1") / Decimal(110_000) * 1_000_000


def test_csv_appends_the_new_columns_without_reordering_the_old_ones() -> None:
    body = to_csv(report([token_row("checkout", "1")]))
    header = next(csv.reader(io.StringIO(body)))

    assert header[:9] == [
        "project",
        "requests",
        "subjects",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "direct_usd",
        "allocated_usd",
    ]
    assert {"reasoning_tokens", "input_usd", "usd_per_1m_tokens"} <= set(header)
