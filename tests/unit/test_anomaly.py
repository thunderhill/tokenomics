from __future__ import annotations

import random
import statistics
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from tokenomics.finops.anomaly import (
    attribute_cause,
    detect,
    robust_scale,
    robust_z_scores,
    seasonal_profile,
)

START = datetime(2026, 8, 1, tzinfo=UTC)


def diurnal_series(
    hours: int = 24 * 21, *, peak: float = 2.0, trough: float = 0.5, seed: int = 11
) -> list[tuple[datetime, Decimal]]:
    """Business-hours-heavy hourly spend, the shape real LLM workloads actually have."""
    rng = random.Random(seed)
    series = []
    for hour in range(hours):
        moment = START + timedelta(hours=hour)
        base = peak if (9 <= moment.hour < 18 and moment.weekday() < 5) else trough
        series.append((moment, Decimal(str(round(base * rng.uniform(0.85, 1.15), 4)))))
    return series


def test_quiet_series_produces_no_anomalies():
    """The regression that matters: normal business hours are not anomalies."""
    assert detect(diurnal_series()) == []


def test_finds_an_injected_spike():
    series = diurnal_series()
    series[300] = (series[300][0], Decimal("45.0"))
    found = detect(series)
    assert len(found) == 1
    assert found[0].bucket == series[300][0]
    assert found[0].observed_usd == Decimal("45.0")


def test_baseline_is_seasonal_not_global():
    """A midday spike must be compared with midday, not with the 3am average."""
    series = diurnal_series()
    series[300] = (series[300][0], Decimal("45.0"))
    anomaly = detect(series)[0]
    global_median = statistics.median([float(v) for _, v in series])
    assert anomaly.baseline_usd > Decimal(str(global_median))
    assert float(anomaly.baseline_usd) > 1.5


def test_trivial_deviations_are_not_reported():
    """A statistically odd but financially irrelevant blip is noise, not an anomaly."""
    series = diurnal_series()
    series[300] = (series[300][0], Decimal("2.9"))  # odd for the hour, but pennies
    assert detect(series) == []


def test_materiality_floor_is_configurable():
    series = diurnal_series()
    series[300] = (series[300][0], Decimal("8.0"))
    assert detect(series, min_deviation_usd=Decimal("100")) == []
    assert len(detect(series, min_deviation_usd=Decimal("1"))) == 1


def test_a_huge_spike_does_not_mask_a_later_one():
    """The reason for median/MAD over mean/stdev.

    One enormous spike inflates the standard deviation enough to hide every later
    spike. A robust scale is essentially unmoved by it.
    """
    series = diurnal_series()
    series[300] = (series[300][0], Decimal("2000.0"))
    series[400] = (series[400][0], Decimal("40.0"))
    buckets = {a.bucket for a in detect(series)}
    assert series[300][0] in buckets
    assert series[400][0] in buckets, "the later, smaller spike was masked"


def test_robust_scale_survives_a_degenerate_mad():
    """When most points are identical the MAD is 0; the fallback must stay robust.

    Neither the standard deviation (inflated by the outliers) nor the spread of the
    nonzero deviations (defined *by* the outliers) is acceptable here.
    """
    values = [2.0] * 60 + [600.0, 45.0]
    scale = robust_scale(values)
    assert scale > 0
    assert scale < statistics.pstdev(values), "scale was inflated by the outliers"

    scores = robust_z_scores(values)
    assert scores[60] > 3, "the huge spike must be flagged"
    assert scores[61] > 3, "the smaller spike must not be masked by the huge one"


def test_zero_spread_yields_zero_scores():
    assert robust_z_scores([5.0] * 10) == [0.0] * 10


def test_too_little_history_reports_nothing():
    assert detect(diurnal_series(hours=5)) == []


def test_seasonal_profile_separates_weekend_from_weekday():
    profile = seasonal_profile([(m, float(v)) for m, v in diurnal_series()])
    assert profile[(12, False)] > profile[(12, True)]  # noon Tue vs noon Sat


def test_attribute_cause_ranks_by_contribution():
    causes = attribute_cause(
        observed={
            ("feature", "summarizer"): Decimal("40"),
            ("feature", "chat"): Decimal("3"),
        },
        baseline={
            ("feature", "summarizer"): Decimal("2"),
            ("feature", "chat"): Decimal("2"),
        },
    )
    assert causes[0].value == "summarizer"
    assert causes[0].share > 0.9
    assert sum(c.share for c in causes) <= 1.0000001


def test_attribute_cause_ignores_shrinking_dimensions():
    causes = attribute_cause(
        observed={("model_key", "a"): Decimal("1")},
        baseline={("model_key", "a"): Decimal("50")},
    )
    assert causes == ()
