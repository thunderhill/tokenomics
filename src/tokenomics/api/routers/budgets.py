"""Budget CRUD, live status, and the evaluation cycle."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from tokenomics.api.deps import Db
from tokenomics.finops import budgets as budget_service
from tokenomics.finops.budgets import Budget, InvalidScopeError, Period
from tokenomics.telemetry import metrics

router = APIRouter(prefix="/api/budgets", tags=["budgets"])


class BudgetIn(BaseModel):
    name: str
    amount_usd: Decimal = Field(gt=0)
    scope: dict[str, Any] = Field(
        default_factory=dict,
        description="Attribution selector, e.g. {'project': 'checkout', 'tags': {'team': 'ml'}}.",
    )
    period: Period = Period.MONTHLY
    rolling_days: int | None = Field(default=None, gt=0)
    thresholds: list[Decimal] = Field(default=[Decimal("0.5"), Decimal("0.8"), Decimal("1.0")])
    webhook_url: str | None = None
    webhook_secret: str | None = None
    enabled: bool = True

    def to_budget(self) -> Budget:
        return Budget(
            name=self.name,
            amount_usd=self.amount_usd,
            scope=self.scope,
            period=self.period,
            rolling_days=self.rolling_days,
            thresholds=tuple(self.thresholds),
            webhook_url=self.webhook_url,
            webhook_secret=self.webhook_secret,
            enabled=self.enabled,
        )


class BudgetOut(BaseModel):
    id: UUID
    name: str
    amount_usd: Decimal
    scope: dict[str, Any]
    period: Period
    rolling_days: int | None
    thresholds: list[Decimal]
    webhook_url: str | None
    #: The secret is never returned; only whether one is configured.
    has_webhook_secret: bool
    enabled: bool

    @classmethod
    def of(cls, budget: Budget) -> BudgetOut:
        assert budget.id is not None
        return cls(
            id=budget.id,
            name=budget.name,
            amount_usd=budget.amount_usd,
            scope=dict(budget.scope),
            period=budget.period,
            rolling_days=budget.rolling_days,
            thresholds=list(budget.thresholds),
            webhook_url=budget.webhook_url,
            has_webhook_secret=bool(budget.webhook_secret),
            enabled=budget.enabled,
        )


class BudgetStatusOut(BaseModel):
    budget: BudgetOut
    period_start: datetime
    since: datetime
    until: datetime
    spend_usd: Decimal
    amount_usd: Decimal
    remaining_usd: Decimal
    utilization: float
    requests: int
    unpriced_events: int
    has_blind_spot: bool
    thresholds_crossed: list[Decimal]


class AlertOut(BaseModel):
    id: int
    budget_id: UUID
    budget_name: str
    period_start: datetime
    threshold: Decimal
    spend_usd: Decimal
    amount_usd: Decimal
    fired_at: datetime
    delivered: bool
    delivery_error: str | None


@router.get("", response_model=list[BudgetOut], summary="List budgets")
def list_budgets(conn: Db, enabled_only: bool = False) -> list[BudgetOut]:
    return [BudgetOut.of(b) for b in budget_service.list_budgets(conn, enabled_only=enabled_only)]


@router.post(
    "", response_model=BudgetOut, status_code=status.HTTP_201_CREATED, summary="Create a budget"
)
def create_budget(conn: Db, payload: BudgetIn) -> BudgetOut:
    try:
        budget = payload.to_budget()
    except (InvalidScopeError, ValueError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return BudgetOut.of(budget_service.create_budget(conn, budget))


@router.get("/alerts", response_model=list[AlertOut], summary="Recently fired alerts")
def alerts(
    conn: Db,
    budget_id: UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[dict[str, Any]]:
    return budget_service.recent_alerts(conn, budget_id=budget_id, limit=limit)


@router.get("/{budget_id}", response_model=BudgetOut, summary="Fetch one budget")
def get_budget(conn: Db, budget_id: UUID) -> BudgetOut:
    return BudgetOut.of(_require(conn, budget_id))


@router.delete("/{budget_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a budget")
def delete_budget(conn: Db, budget_id: UUID) -> None:
    if not budget_service.delete_budget(conn, budget_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such budget")


@router.get(
    "/{budget_id}/status",
    response_model=BudgetStatusOut,
    summary="Live burn-down for the current period",
)
def budget_status(conn: Db, budget_id: UUID) -> BudgetStatusOut:
    return _status_out(budget_service.evaluate(conn, _require(conn, budget_id)))


@router.post(
    "/evaluate",
    response_model=list[AlertOut],
    summary="Evaluate every budget and fire newly crossed thresholds",
    description=(
        "Idempotent by construction: a threshold already fired this period is skipped, so "
        "this is safe to call from a cron as often as you like."
    ),
)
def evaluate_all(conn: Db) -> list[dict[str, Any]]:
    fired = budget_service.run_budget_cycle(conn)
    for budget in budget_service.list_budgets(conn):
        status_ = budget_service.evaluate(conn, budget)
        metrics.budget_utilization.labels(budget=budget.name).set(float(status_.utilization))
    return [
        {
            "id": alert.id,
            "budget_id": alert.budget.id,
            "budget_name": alert.budget.name,
            "period_start": datetime.combine(alert.period_start, datetime.min.time(), UTC),
            "threshold": alert.threshold,
            "spend_usd": alert.spend_usd,
            "amount_usd": alert.amount_usd,
            "fired_at": alert.fired_at,
            "delivered": False,
            "delivery_error": None,
        }
        for alert in fired
    ]


def _require(conn: Db, budget_id: UUID) -> Budget:
    budget = budget_service.get_budget(conn, budget_id)
    if budget is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such budget")
    return budget


def _status_out(status_: budget_service.BudgetStatus) -> BudgetStatusOut:
    return BudgetStatusOut(
        budget=BudgetOut.of(status_.budget),
        period_start=datetime.combine(status_.period_start, datetime.min.time(), UTC),
        since=status_.since,
        until=status_.until,
        spend_usd=status_.spend_usd,
        amount_usd=status_.amount_usd,
        remaining_usd=status_.remaining_usd,
        utilization=float(status_.utilization),
        requests=status_.requests,
        unpriced_events=status_.unpriced_events,
        has_blind_spot=status_.has_blind_spot,
        thresholds_crossed=list(status_.crossed()),
    )
