"""Month-end spend forecasting.

Linear trend with multiplicative day-of-week seasonality. LLM spend is strongly weekly
-- weekday traffic routinely runs several times weekend traffic -- so a plain run-rate
extrapolation lands badly depending on which day of the week you happen to ask.

With less than two weeks of history the model falls back to a flat mean. Fitting a trend
to a handful of noisy days produces confident nonsense, and a forecast nobody can trust
is worse than an obviously simple one.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal

MIN_DAYS_FOR_TREND = 14
_DAYS_IN_WEEK = 7


@dataclass(frozen=True, slots=True)
class Forecast:
    """A month-end projection with an uncertainty band."""

    month_to_date_usd: Decimal
    projected_month_end_usd: Decimal
    lower_usd: Decimal
    upper_usd: Decimal
    daily_run_rate_usd: Decimal
    days_observed: int
    days_remaining: int
    method: str
    day_of_week_factors: tuple[float, ...] = ()


def _month_end(day: date) -> date:
    first_next = date(day.year + 1, 1, 1) if day.month == 12 else date(day.year, day.month + 1, 1)
    return first_next - timedelta(days=1)


def _least_squares(values: list[float]) -> tuple[float, float]:
    """Fit ``y = intercept + slope * t`` over ``t = 0..n-1``."""
    n = len(values)
    mean_t = (n - 1) / 2
    mean_y = sum(values) / n
    variance = sum((t - mean_t) ** 2 for t in range(n))
    if variance == 0:
        return mean_y, 0.0
    covariance = sum((t - mean_t) * (y - mean_y) for t, y in enumerate(values))
    slope = covariance / variance
    return mean_y - slope * mean_t, slope


def _dow_factors(days: list[date], values: list[float], baseline: list[float]) -> list[float]:
    """Multiplicative day-of-week factors, normalized to average 1."""
    ratios: dict[int, list[float]] = {d: [] for d in range(_DAYS_IN_WEEK)}
    for day, actual, expected in zip(days, values, baseline, strict=True):
        if expected > 0:
            ratios[day.weekday()].append(actual / expected)

    factors = [statistics.median(ratios[d]) if ratios[d] else 1.0 for d in range(_DAYS_IN_WEEK)]
    average = sum(factors) / _DAYS_IN_WEEK
    if average <= 0:
        return [1.0] * _DAYS_IN_WEEK
    return [factor / average for factor in factors]


def forecast_month_end(
    series: list[tuple[datetime | date, Decimal]],
    *,
    today: date | None = None,
) -> Forecast:
    """Project calendar-month spend from a daily series.

    ``series`` need not be dense; missing days are treated as zero-spend days, which is
    what they are.
    """
    observations = {
        (point.date() if isinstance(point, datetime) else point): float(value)
        for point, value in series
    }
    today = today or (max(observations) if observations else date.today())

    month_start = date(today.year, today.month, 1)
    month_end = _month_end(today)
    in_month = {day: v for day, v in observations.items() if month_start <= day <= today}

    days_remaining = (month_end - today).days
    month_to_date = Decimal(str(sum(in_month.values())))

    # Dense day-by-day history (absent days really are zero-spend days).
    history_days = sorted(observations)
    if not history_days:
        zero = Decimal(0)
        return Forecast(zero, zero, zero, zero, zero, 0, days_remaining, "no-data")

    span = [
        history_days[0] + timedelta(days=offset)
        for offset in range((history_days[-1] - history_days[0]).days + 1)
    ]
    values = [observations.get(day, 0.0) for day in span]

    if len(values) < MIN_DAYS_FOR_TREND:
        run_rate = sum(values) / len(values)
        projected = month_to_date + Decimal(str(run_rate * days_remaining))
        deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
        band = Decimal(str(deviation * (days_remaining**0.5)))
        return Forecast(
            month_to_date_usd=month_to_date,
            projected_month_end_usd=projected,
            lower_usd=max(projected - band, month_to_date),
            upper_usd=projected + band,
            daily_run_rate_usd=Decimal(str(run_rate)),
            days_observed=len(values),
            days_remaining=days_remaining,
            method="mean-fallback",
        )

    intercept, slope = _least_squares(values)
    trend = [max(intercept + slope * t, 0.0) for t in range(len(values))]
    factors = _dow_factors(span, values, trend)

    fitted = [trend[i] * factors[span[i].weekday()] for i in range(len(values))]
    residuals = [actual - predicted for actual, predicted in zip(values, fitted, strict=True)]
    sigma = statistics.pstdev(residuals) if len(residuals) > 1 else 0.0

    projected_remainder = 0.0
    for offset in range(1, days_remaining + 1):
        future_day = today + timedelta(days=offset)
        t = len(values) - 1 + offset
        projected_remainder += max(intercept + slope * t, 0.0) * factors[future_day.weekday()]

    projected = month_to_date + Decimal(str(projected_remainder))
    band = Decimal(str(sigma * (days_remaining**0.5)))

    return Forecast(
        month_to_date_usd=month_to_date,
        projected_month_end_usd=projected,
        lower_usd=max(projected - band, month_to_date),
        upper_usd=projected + band,
        daily_run_rate_usd=Decimal(str(sum(values[-7:]) / min(7, len(values)))),
        days_observed=len(values),
        days_remaining=days_remaining,
        method="trend+dow",
        day_of_week_factors=tuple(factors),
    )
