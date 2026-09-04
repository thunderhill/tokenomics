"""Chargeback, showback and unit-economics reporting.

Showback tells a team what it spent. Chargeback *bills* it, which raises a question
showback never has to answer: what happens to spend that no team owns? Events arrive
attributed to ``unknown`` more often than anyone likes -- a cron job nobody tagged, a
migration script, an SDK integration that shipped without ``project=``. Two policies
are supported and the choice is recorded in the report:

``show``    keep unallocated spend as its own line. Honest, and the line's size is a
            useful nag. This is the default.
``spread``  allocate it pro-rata across the owning lines, the way shared infrastructure
            cost is normally handled. The allocation is penny-exact: rounding residue
            goes to the largest consumer rather than quietly disappearing.

Unpriced events are carried through every line. A chargeback report whose numbers are
missing a model is a report that will be disputed, so the gap travels with the money.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Any

import psycopg

from tokenomics.finops.tokens import PER_MILLION
from tokenomics.storage.queries import Filters, spend_breakdown

#: Label used for spend that carries no owning attribution.
UNALLOCATED = "unallocated"

#: Values that mean "nobody claimed this".
_UNOWNED = frozenset({None, "", "unknown"})

CENT = Decimal("0.01")


class Allocation(StrEnum):
    SHOW = "show"
    SPREAD = "spread"


@dataclass(frozen=True, slots=True)
class ReportLine:
    """One billable entity's slice of the bill."""

    key: tuple[str, ...]
    requests: int
    subjects: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    direct_usd: Decimal
    allocated_usd: Decimal
    unpriced_events: int
    # Everything below defaults, so a line can still be built from the money alone --
    # which is all `spread()` and its tests need.
    reasoning_tokens: int = 0
    billable_input_tokens: int = 0
    # Per-component spend, so a disputed line can be argued about at the level the
    # provider actually bills at. These sum to `direct_usd`, never to `total_usd`:
    # allocated spend is somebody else's tokens.
    input_usd: Decimal = Decimal(0)
    output_usd: Decimal = Decimal(0)
    cache_read_usd: Decimal = Decimal(0)
    cache_write_usd: Decimal = Decimal(0)
    reasoning_usd: Decimal = Decimal(0)

    @property
    def name(self) -> str:
        return " / ".join(self.key)

    @property
    def total_usd(self) -> Decimal:
        return self.direct_usd + self.allocated_usd

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_per_request(self) -> Decimal | None:
        return self.total_usd / self.requests if self.requests else None

    @property
    def cost_per_subject(self) -> Decimal | None:
        return self.total_usd / self.subjects if self.subjects else None

    @property
    def cost_per_1k_tokens(self) -> Decimal | None:
        if not self.total_tokens:
            return None
        return self.total_usd / Decimal(self.total_tokens) * 1000

    @property
    def usd_per_1m_tokens(self) -> Decimal | None:
        """Blended effective rate, in the unit providers quote prices in."""
        if not self.total_tokens:
            return None
        return self.total_usd / Decimal(self.total_tokens) * PER_MILLION

    @property
    def cache_hit_rate(self) -> float | None:
        """Share of input tokens served from cache -- the main lever on input cost."""
        if not self.input_tokens:
            return None
        return self.cache_read_tokens / self.input_tokens

    @property
    def reasoning_share(self) -> float | None:
        """Share of output tokens spent thinking rather than answering."""
        if not self.output_tokens:
            return None
        return self.reasoning_tokens / self.output_tokens

    @property
    def output_cost_share(self) -> float | None:
        """Share of this line's own spend that went on output tokens.

        Usually the largest single number on the report and the least expected one:
        output is a small share of the volume and a large share of the bill.
        """
        if not self.direct_usd:
            return None
        return float(self.output_usd / self.direct_usd)


@dataclass(frozen=True, slots=True)
class Report:
    kind: str
    group_by: tuple[str, ...]
    since: datetime
    until: datetime
    lines: tuple[ReportLine, ...]
    unallocated_usd: Decimal
    allocation: Allocation
    currency: str = "USD"

    @property
    def total_usd(self) -> Decimal:
        # Under `show` the unallocated spend is one of the lines; under `spread` it has
        # been folded into them. Either way the lines are the whole bill -- adding
        # `unallocated_usd` on top would double-count it.
        return sum((line.total_usd for line in self.lines), Decimal(0))

    @property
    def unpriced_events(self) -> int:
        return sum(line.unpriced_events for line in self.lines)

    def share(self, line: ReportLine) -> float:
        total = self.total_usd
        return float(line.total_usd / total) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "group_by": list(self.group_by),
            "since": self.since.isoformat(),
            "until": self.until.isoformat(),
            "currency": self.currency,
            "allocation": str(self.allocation),
            "total_usd": str(self.total_usd),
            "unallocated_usd": str(self.unallocated_usd),
            "unpriced_events": self.unpriced_events,
            "lines": [
                {
                    **dict(zip(self.group_by, line.key, strict=True)),
                    "requests": line.requests,
                    "subjects": line.subjects,
                    "input_tokens": line.input_tokens,
                    "output_tokens": line.output_tokens,
                    "cache_read_tokens": line.cache_read_tokens,
                    "cache_write_tokens": line.cache_write_tokens,
                    "reasoning_tokens": line.reasoning_tokens,
                    "billable_input_tokens": line.billable_input_tokens,
                    "total_tokens": line.total_tokens,
                    "input_usd": str(line.input_usd),
                    "output_usd": str(line.output_usd),
                    "cache_read_usd": str(line.cache_read_usd),
                    "cache_write_usd": str(line.cache_write_usd),
                    "reasoning_usd": str(line.reasoning_usd),
                    "direct_usd": str(line.direct_usd),
                    "allocated_usd": str(line.allocated_usd),
                    "total_usd": str(line.total_usd),
                    "share": round(self.share(line), 6),
                    "cost_per_request": _opt(line.cost_per_request),
                    "cost_per_subject": _opt(line.cost_per_subject),
                    "cost_per_1k_tokens": _opt(line.cost_per_1k_tokens),
                    "usd_per_1m_tokens": _opt(line.usd_per_1m_tokens),
                    "cache_hit_rate": line.cache_hit_rate,
                    "reasoning_share": line.reasoning_share,
                    "output_cost_share": line.output_cost_share,
                    "unpriced_events": line.unpriced_events,
                }
                for line in self.lines
            ],
        }


def _opt(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _is_unowned(value: Any) -> bool:
    return value in _UNOWNED


def _line_from_row(row: dict[str, Any], group_by: tuple[str, ...]) -> ReportLine:
    return ReportLine(
        key=tuple(str(row.get(d) or UNALLOCATED) for d in group_by),
        requests=int(row.get("requests") or 0),
        subjects=int(row.get("subjects") or 0),
        input_tokens=int(row.get("input_tokens") or 0),
        output_tokens=int(row.get("output_tokens") or 0),
        cache_read_tokens=int(row.get("cache_read_tokens") or 0),
        cache_write_tokens=int(row.get("cache_write_tokens") or 0),
        reasoning_tokens=int(row.get("reasoning_tokens") or 0),
        billable_input_tokens=int(row.get("billable_input_tokens") or 0),
        direct_usd=Decimal(row.get("cost_usd") or 0),
        allocated_usd=Decimal(0),
        unpriced_events=int(row.get("unpriced_events") or 0),
        input_usd=Decimal(row.get("input_usd") or 0),
        output_usd=Decimal(row.get("output_usd") or 0),
        cache_read_usd=Decimal(row.get("cache_read_usd") or 0),
        cache_write_usd=Decimal(row.get("cache_write_usd") or 0),
        reasoning_usd=Decimal(row.get("reasoning_usd") or 0),
    )


def spread(lines: list[ReportLine], amount: Decimal) -> list[ReportLine]:
    """Allocate ``amount`` across ``lines`` pro-rata by direct spend.

    Only the money moves: the unowned requests and tokens are not merged into the
    owning lines, so their per-request figures stay a true measure of their own traffic
    carrying its share of the shared bill.

    Falls back to request share when nobody has any priced spend to weight by. Rounding
    residue is given to the largest line so the allocated total is exactly ``amount``
    -- a chargeback report that does not add up gets rejected by finance, rightly.
    """
    if not lines or amount == 0:
        return lines

    weights = [line.direct_usd for line in lines]
    if sum(weights) == 0:
        weights = [Decimal(line.requests) for line in lines]
    total_weight = sum(weights)
    if total_weight == 0:
        return lines

    shares = [
        (amount * weight / total_weight).quantize(CENT, rounding=ROUND_HALF_UP)
        for weight in weights
    ]
    residue = amount.quantize(CENT, rounding=ROUND_HALF_UP) - sum(shares)
    if residue:
        biggest = max(range(len(shares)), key=lambda i: weights[i])
        shares[biggest] += residue

    return [
        ReportLine(**{**_as_dict(line), "allocated_usd": share})
        for line, share in zip(lines, shares, strict=True)
    ]


def _as_dict(line: ReportLine) -> dict[str, Any]:
    return {f: getattr(line, f) for f in ReportLine.__dataclass_fields__}


def build_report(
    rows: list[dict[str, Any]],
    *,
    group_by: tuple[str, ...],
    since: datetime,
    until: datetime,
    allocation: Allocation = Allocation.SHOW,
    kind: str = "chargeback",
) -> Report:
    """Turn aggregated rows into a report. Pure: the query lives in :func:`chargeback`."""
    owned: list[ReportLine] = []
    unallocated = Decimal(0)
    unallocated_line: ReportLine | None = None

    for row in rows:
        line = _line_from_row(row, group_by)
        # Ownership is decided by the *first* dimension: that is who gets the invoice.
        if _is_unowned(row.get(group_by[0])):
            # NULL and the literal 'unknown' are the same fact, so normalize the label
            # before merging or they land on two separate lines.
            line = ReportLine(**{**_as_dict(line), "key": (UNALLOCATED, *line.key[1:])})
            unallocated += line.direct_usd
            unallocated_line = _merge(unallocated_line, line)
        else:
            owned.append(line)

    owned.sort(key=lambda line: line.direct_usd, reverse=True)

    # Spreading with nobody to spread onto would delete the money from the report, so
    # fall back to showing it. The report records which policy actually applied.
    effective = allocation if (allocation is Allocation.SPREAD and owned) else Allocation.SHOW
    if effective is Allocation.SPREAD:
        lines = spread(owned, unallocated)
    else:
        lines = owned + ([unallocated_line] if unallocated_line else [])

    return Report(
        kind=kind,
        group_by=group_by,
        since=since,
        until=until,
        lines=tuple(lines),
        unallocated_usd=unallocated,
        allocation=effective,
    )


def chargeback(
    conn: psycopg.Connection,
    *,
    filters: Filters,
    group_by: tuple[str, ...] = ("project",),
    allocation: Allocation = Allocation.SHOW,
    kind: str = "chargeback",
    limit: int = 500,
) -> Report:
    """Per-owner cost report over a period."""
    rows = spend_breakdown(conn, filters=filters, group_by=list(group_by), limit=limit)
    return build_report(
        rows,
        group_by=group_by,
        since=filters.since,
        until=filters.until,
        allocation=allocation,
        kind=kind,
    )


def showback(conn: psycopg.Connection, **kwargs: Any) -> Report:
    """Same numbers as chargeback, without moving anyone's budget."""
    kwargs.setdefault("allocation", Allocation.SHOW)
    return chargeback(conn, kind="showback", **kwargs)


def _merge(existing: ReportLine | None, new: ReportLine) -> ReportLine:
    """Fold several unowned rows into one line (e.g. NULL and 'unknown')."""
    if existing is None:
        return new
    return ReportLine(
        key=existing.key,
        requests=existing.requests + new.requests,
        subjects=existing.subjects + new.subjects,
        input_tokens=existing.input_tokens + new.input_tokens,
        output_tokens=existing.output_tokens + new.output_tokens,
        cache_read_tokens=existing.cache_read_tokens + new.cache_read_tokens,
        cache_write_tokens=existing.cache_write_tokens + new.cache_write_tokens,
        reasoning_tokens=existing.reasoning_tokens + new.reasoning_tokens,
        billable_input_tokens=existing.billable_input_tokens + new.billable_input_tokens,
        direct_usd=existing.direct_usd + new.direct_usd,
        allocated_usd=existing.allocated_usd + new.allocated_usd,
        unpriced_events=existing.unpriced_events + new.unpriced_events,
        input_usd=existing.input_usd + new.input_usd,
        output_usd=existing.output_usd + new.output_usd,
        cache_read_usd=existing.cache_read_usd + new.cache_read_usd,
        cache_write_usd=existing.cache_write_usd + new.cache_write_usd,
        reasoning_usd=existing.reasoning_usd + new.reasoning_usd,
    )


#: Appended to, never reordered: finance imports this by column position often enough
#: that shuffling it would silently corrupt somebody's spreadsheet.
_CSV_COLUMNS = (
    "requests",
    "subjects",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "direct_usd",
    "allocated_usd",
    "total_usd",
    "share",
    "cost_per_request",
    "cost_per_subject",
    "cost_per_1k_tokens",
    "cache_hit_rate",
    "unpriced_events",
    "reasoning_tokens",
    "billable_input_tokens",
    "total_tokens",
    "input_usd",
    "output_usd",
    "cache_read_usd",
    "cache_write_usd",
    "reasoning_usd",
    "usd_per_1m_tokens",
    "reasoning_share",
    "output_cost_share",
)


def to_csv(report: Report) -> str:
    """Flat CSV, one row per line item -- what finance actually asks for."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([*report.group_by, *_CSV_COLUMNS])
    for item in report.to_dict()["lines"]:
        writer.writerow(
            [item.get(d, "") for d in report.group_by]
            + [_csv_value(item.get(column)) for column in _CSV_COLUMNS]
        )
    return buffer.getvalue()


def _csv_value(value: Any) -> Any:
    return "" if value is None else value
