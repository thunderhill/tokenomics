"""Forecast, anomalies and the what-if simulator."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from tokenomics.api.deps import Db, Engine, FilterQuery
from tokenomics.finops import anomaly as anomaly_service
from tokenomics.finops import forecast as forecast_service
from tokenomics.finops import whatif as whatif_service
from tokenomics.finops.whatif import QualityAssumption
from tokenomics.storage import queries
from tokenomics.telemetry import metrics

router = APIRouter(prefix="/api", tags=["finops"])


class ForecastOut(BaseModel):
    month_to_date_usd: Decimal
    projected_month_end_usd: Decimal
    lower_usd: Decimal
    upper_usd: Decimal
    daily_run_rate_usd: Decimal
    days_observed: int
    days_remaining: int
    method: str = Field(description="'trend+dow', 'mean-fallback' or 'no-data'.")
    day_of_week_factors: list[float]


class CauseOut(BaseModel):
    dimension: str
    value: str
    delta_usd: Decimal
    share: float


class AnomalyOut(BaseModel):
    bucket: datetime
    observed_usd: Decimal
    baseline_usd: Decimal
    deviation_usd: Decimal
    score: float
    multiple: float
    probable_cause: list[CauseOut]


class SimulationOut(BaseModel):
    target_model: str
    resolved_model_key: str | None
    snapshot_id: str
    requests: int
    repriced_requests: int
    unpriceable_requests: int
    baseline_usd: Decimal
    projected_usd: Decimal
    delta_usd: Decimal
    delta_pct: float | None
    refolded_cache_tokens: int
    scale_factor: float
    is_complete: bool
    warnings: list[str]
    quality: dict[str, Any]


class WhatIfIn(BaseModel):
    targets: list[str] = Field(min_length=1, description="Model keys to reprice against.")
    keep_service_tier: bool = True
    max_shapes: int = Field(default=whatif_service.DEFAULT_MAX_SHAPES, ge=100)
    quality: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional, self-reported. Tokenomics never infers answer quality.",
    )


@router.get("/forecast", response_model=ForecastOut, summary="Month-end spend projection")
def forecast(
    conn: Db,
    filters: FilterQuery,
    today: datetime | None = None,
) -> ForecastOut:
    series = queries.daily_spend(conn, filters=filters)
    result = forecast_service.forecast_month_end(
        list(series), today=(today or datetime.now(UTC)).date()
    )
    return ForecastOut(
        month_to_date_usd=result.month_to_date_usd,
        projected_month_end_usd=result.projected_month_end_usd,
        lower_usd=result.lower_usd,
        upper_usd=result.upper_usd,
        daily_run_rate_usd=result.daily_run_rate_usd,
        days_observed=result.days_observed,
        days_remaining=result.days_remaining,
        method=result.method,
        day_of_week_factors=list(result.day_of_week_factors),
    )


@router.get(
    "/anomalies",
    response_model=list[AnomalyOut],
    summary="Spend anomalies with a probable cause",
    description=(
        "Hourly spend is deseasonalized against an hour-of-day x day-type profile, then "
        "scored with a median/MAD z-score so one huge spike cannot mask the next one."
    ),
)
def anomalies(
    conn: Db,
    filters: FilterQuery,
    threshold: Annotated[float, Query(ge=1.0, le=10.0)] = anomaly_service.DEFAULT_THRESHOLD,
    min_deviation_usd: Decimal = anomaly_service.DEFAULT_MIN_DEVIATION_USD,
    persist: Annotated[bool, Query(description="Store the results in the anomaly feed.")] = False,
) -> list[AnomalyOut]:
    found = anomaly_service.scan(
        conn, filters=filters, threshold=threshold, min_deviation_usd=min_deviation_usd
    )
    if persist and found:
        anomaly_service.persist(conn, found)
        metrics.anomalies_detected.inc(len(found))
    return [
        AnomalyOut(
            bucket=item.bucket,
            observed_usd=item.observed_usd,
            baseline_usd=item.baseline_usd,
            deviation_usd=item.deviation_usd,
            score=item.score,
            multiple=item.multiple,
            probable_cause=[
                CauseOut(
                    dimension=cause.dimension,
                    value=cause.value,
                    delta_usd=cause.delta_usd,
                    share=cause.share,
                )
                for cause in item.probable_cause
            ],
        )
        for item in found
    ]


@router.post(
    "/whatif",
    response_model=list[SimulationOut],
    summary="Reprice historical traffic against other models",
    description=(
        "Replays stored token vectors -- not aggregate totals -- so per-request "
        "long-context pricing stays correct. If the target model does not price prompt "
        "caching, cached tokens are re-billed at its full input rate and the count is "
        "reported, because that is a prompt-engineering finding, not a rounding error."
    ),
)
def whatif(
    conn: Db, engine: Engine, filters: FilterQuery, payload: WhatIfIn
) -> list[SimulationOut]:
    samples, scale = whatif_service.load_samples(
        conn, filters=filters, max_shapes=payload.max_shapes
    )
    results = whatif_service.compare(
        engine,
        samples,
        targets=payload.targets,
        keep_service_tier=payload.keep_service_tier,
        scale_factor=scale,
        quality=QualityAssumption(**payload.quality) if payload.quality else None,
    )
    return [
        SimulationOut(
            target_model=result.target_model,
            resolved_model_key=result.resolved_model_key,
            snapshot_id=result.snapshot_id,
            requests=result.requests,
            repriced_requests=result.repriced_requests,
            unpriceable_requests=result.unpriceable_requests,
            baseline_usd=result.baseline_usd,
            projected_usd=result.projected_usd,
            delta_usd=result.delta_usd,
            delta_pct=result.delta_pct,
            refolded_cache_tokens=result.refolded_cache_tokens,
            scale_factor=result.scale_factor,
            is_complete=result.is_complete,
            warnings=list(result.warnings),
            quality=asdict(result.quality),
        )
        for result in results
    ]
