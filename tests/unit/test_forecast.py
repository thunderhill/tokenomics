from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import Decimal

from tokenomics.finops.forecast import MIN_DAYS_FOR_TREND, forecast_month_end


def build_series(
    days: int,
    *,
    base: float = 100.0,
    growth: float = 0.0,
    weekend_factor: float = 1.0,
    start: date = date(2026, 8, 1),
    jitter: float = 0.0,
    seed: int = 3,
) -> list[tuple[date, Decimal]]:
    rng = random.Random(seed)
    series = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        value = (base + growth * offset) * (weekend_factor if day.weekday() >= 5 else 1.0)
        if jitter:
            value *= rng.uniform(1 - jitter, 1 + jitter)
        series.append((day, Decimal(str(round(value, 4)))))
    return series


def test_flat_series_projects_flat():
    series = build_series(20, base=10.0)
    result = forecast_month_end(series, today=date(2026, 8, 20))
    # 20 days elapsed at 10/day, 11 days remaining
    assert result.projected_month_end_usd == Decimal("310.0000")


def test_short_history_uses_mean_fallback():
    """Fitting a trend to a few noisy days produces confident nonsense."""
    result = forecast_month_end(build_series(5, base=10.0), today=date(2026, 8, 5))
    assert result.method == "mean-fallback"


def test_switches_to_trend_at_the_threshold():
    series = build_series(MIN_DAYS_FOR_TREND, base=10.0)
    result = forecast_month_end(series, today=date(2026, 8, MIN_DAYS_FOR_TREND))
    assert result.method == "trend+dow"


def test_learns_weekend_seasonality():
    """Weekday factors must exceed weekend factors for a weekday-heavy workload."""
    series = build_series(21, base=100.0, weekend_factor=0.3)
    result = forecast_month_end(series, today=date(2026, 8, 21))
    weekday = result.day_of_week_factors[:5]
    weekend = result.day_of_week_factors[5:]
    assert min(weekday) > max(weekend)


def test_growth_is_projected_forward():
    flat = forecast_month_end(build_series(20, base=100.0), today=date(2026, 8, 20))
    rising = forecast_month_end(build_series(20, base=100.0, growth=5.0), today=date(2026, 8, 20))
    assert rising.projected_month_end_usd > flat.projected_month_end_usd


def test_projection_never_falls_below_money_already_spent():
    series = build_series(20, base=100.0, growth=-4.5)
    result = forecast_month_end(series, today=date(2026, 8, 20))
    assert result.projected_month_end_usd >= result.month_to_date_usd
    assert result.lower_usd >= result.month_to_date_usd


def test_interval_brackets_the_projection():
    series = build_series(20, base=100.0, jitter=0.3)
    result = forecast_month_end(series, today=date(2026, 8, 20))
    assert result.lower_usd <= result.projected_month_end_usd <= result.upper_usd


def test_noisier_history_widens_the_interval():
    calm = forecast_month_end(build_series(20, base=100.0, jitter=0.02), today=date(2026, 8, 20))
    wild = forecast_month_end(build_series(20, base=100.0, jitter=0.6), today=date(2026, 8, 20))
    calm_band = calm.upper_usd - calm.lower_usd
    wild_band = wild.upper_usd - wild.lower_usd
    assert wild_band > calm_band


def test_empty_series_is_handled():
    result = forecast_month_end([], today=date(2026, 8, 20))
    assert result.method == "no-data"
    assert result.projected_month_end_usd == Decimal(0)


def test_last_day_of_month_has_nothing_left_to_project():
    series = build_series(31, base=10.0)
    result = forecast_month_end(series, today=date(2026, 8, 31))
    assert result.days_remaining == 0
    assert result.projected_month_end_usd == result.month_to_date_usd
