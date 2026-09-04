"""Chargeback and showback exports."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query, Response
from pydantic import BaseModel

from tokenomics.api.deps import Db, FilterQuery, parse_group_by
from tokenomics.finops import reports as report_service
from tokenomics.finops.reports import Allocation

router = APIRouter(prefix="/api/reports", tags=["reports"])


class ReportLineOut(BaseModel):
    dimensions: dict[str, str]
    requests: int
    subjects: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    billable_input_tokens: int
    total_tokens: int
    direct_usd: Decimal
    allocated_usd: Decimal
    total_usd: Decimal
    # Per-component spend. Sums to `direct_usd`; allocated spend has no tokens of
    # its own, so it is deliberately outside this split.
    input_usd: Decimal
    output_usd: Decimal
    cache_read_usd: Decimal
    cache_write_usd: Decimal
    reasoning_usd: Decimal
    share: float
    cost_per_request: Decimal | None
    cost_per_subject: Decimal | None
    cost_per_1k_tokens: Decimal | None
    usd_per_1m_tokens: Decimal | None
    cache_hit_rate: float | None
    reasoning_share: float | None
    output_cost_share: float | None
    unpriced_events: int


class ReportOut(BaseModel):
    kind: str
    group_by: list[str]
    since: str
    until: str
    currency: str
    allocation: str
    total_usd: Decimal
    unallocated_usd: Decimal
    unpriced_events: int
    lines: list[ReportLineOut]


def _to_out(payload: dict[str, Any]) -> ReportOut:
    group_by = payload["group_by"]
    lines = []
    for line in payload["lines"]:
        dimensions = {d: line.pop(d) for d in group_by}
        lines.append(ReportLineOut(dimensions=dimensions, **line))
    return ReportOut(**{**payload, "lines": lines})


@router.get(
    "/chargeback",
    summary="Per-owner cost report",
    description=(
        "``allocation=spread`` distributes unattributed spend across the owning lines "
        "pro-rata, to the cent. ``allocation=show`` (the default) keeps it as its own "
        "line, which is usually the more useful nag."
    ),
    response_model=None,
)
def chargeback(
    conn: Db,
    filters: FilterQuery,
    group_by: Annotated[list[str] | None, Query()] = None,
    allocation: Allocation = Allocation.SHOW,
    kind: Literal["chargeback", "showback"] = "chargeback",
    # `format` is the natural query-parameter name, so alias it rather than
    # shadowing the builtin in Python.
    output_format: Annotated[Literal["json", "csv"], Query(alias="format")] = "json",
) -> ReportOut | Response:
    dimensions = tuple(parse_group_by(group_by) or ["project"])
    report = report_service.chargeback(
        conn, filters=filters, group_by=dimensions, allocation=allocation, kind=kind
    )
    if output_format == "csv":
        return Response(
            report_service.to_csv(report),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{kind}-{dimensions[0]}.csv"'},
        )
    return _to_out(report.to_dict())
