"""Spend anomaly detection.

Uses a **robust** z-score (median / MAD) rather than mean / standard deviation. The
reason is specific to this domain: a runaway agent or a bad prompt deploy produces one
enormous hourly spike, and that single spike inflates the standard deviation so much
that it (a) fails to flag itself and (b) masks every subsequent spike for as long as it
stays in the window. The median and MAD are essentially unmoved by a small number of
extreme points, so the spike stays visible.

The baseline is same-hour-of-week where enough history exists, because LLM spend has
strong daily and weekly shape -- 3am Sunday and 3pm Tuesday are not comparable.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from tokenomics.storage import queries
from tokenomics.storage.queries import Filters

#: Scales MAD to be a consistent estimator of sigma for normally distributed data.
MAD_TO_SIGMA = 1.4826

#: When a series is perfectly flat (MAD == 0), a move of this fraction off the median
#: counts as one standard deviation.
FLAT_SERIES_FRACTION = 0.1

DEFAULT_THRESHOLD = 3.0
MIN_OBSERVATIONS = 12

#: A seasonal profile bucket needs this many samples before it is trusted.
MIN_SAMPLES_PER_BUCKET = 3

#: A statistically striking change that costs nothing is not a FinOps anomaly. These
#: floors keep the feed to things actually worth a human's attention.
DEFAULT_MIN_DEVIATION_USD = Decimal("1.00")
DEFAULT_MIN_MULTIPLE = 1.5


@dataclass(frozen=True, slots=True)
class Cause:
    """A dimension value and how much of the deviation it accounts for."""

    dimension: str
    value: str
    delta_usd: Decimal
    share: float


@dataclass(frozen=True, slots=True)
class Anomaly:
    bucket: datetime
    observed_usd: Decimal
    baseline_usd: Decimal
    deviation_usd: Decimal
    score: float
    probable_cause: tuple[Cause, ...] = field(default_factory=tuple)

    @property
    def multiple(self) -> float:
        if self.baseline_usd == 0:
            return float("inf")
        return float(self.observed_usd / self.baseline_usd)


def robust_scale(values: list[float]) -> float:
    """A robust estimate of spread, degrading sensibly when the MAD is degenerate."""
    median = statistics.median(values)
    deviations = [abs(v - median) for v in values]
    mad = statistics.median(deviations)
    if mad > 0:
        return mad * MAD_TO_SIGMA

    # More than half the points are identical, so the MAD is zero. Two tempting fallbacks
    # are both wrong: the standard deviation surrenders the robustness we came for (one
    # huge outlier inflates it and masks everything after), and the spread of the
    # *nonzero* deviations is set by the outliers themselves, which is worse still.
    #
    # A series this flat genuinely has no measurable spread, so significance becomes
    # relative: treat a FLAT_SERIES_FRACTION move off the median as one sigma.
    if median != 0:
        return abs(median) * FLAT_SERIES_FRACTION
    positive = [d for d in deviations if d > 0]
    return min(positive) if positive else 0.0


def robust_z_scores(values: list[float]) -> list[float]:
    """Median/MAD z-scores. Zero-spread input yields zero scores, not infinities."""
    if len(values) < 2:
        return [0.0] * len(values)
    median = statistics.median(values)
    scale = robust_scale(values)
    if scale == 0:
        return [0.0] * len(values)
    return [(v - median) / scale for v in values]


def _profile_key(moment: datetime) -> tuple[int, bool]:
    """Season bucket: hour of day, split by weekday vs weekend."""
    return moment.hour, moment.weekday() >= 5


def seasonal_profile(series: list[tuple[datetime, float]]) -> dict[tuple[int, bool], float]:
    """Median spend per (hour-of-day, weekday/weekend) bucket.

    LLM spend has strong daily *and* weekly shape, so a single global baseline flags
    every ordinary business hour as a spike. Bucketing by hour-of-day x day-type keeps
    the shape while needing far less history than a full 168-bucket hour-of-week
    profile: two weeks already gives ~10 samples per weekday-hour.
    """
    groups: dict[tuple[int, bool], list[float]] = {}
    for moment, value in series:
        groups.setdefault(_profile_key(moment), []).append(value)

    overall = statistics.median([v for _, v in series]) if series else 0.0
    return {
        key: (statistics.median(values) if len(values) >= MIN_SAMPLES_PER_BUCKET else overall)
        for key, values in groups.items()
    }


def detect(
    series: list[tuple[datetime, Decimal]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_observations: int = MIN_OBSERVATIONS,
    min_deviation_usd: Decimal = DEFAULT_MIN_DEVIATION_USD,
    min_multiple: float = DEFAULT_MIN_MULTIPLE,
) -> list[Anomaly]:
    """Flag hourly buckets whose spend deviates sharply *and materially* from baseline.

    The series is deseasonalized against an hour-of-day x day-type profile first, so the
    z-score measures "unusual for this hour" rather than "bigger than the daily average".
    A bucket must then clear three independent bars: a robust z-score above ``threshold``,
    an absolute increase of at least ``min_deviation_usd``, and a relative increase of at
    least ``min_multiple``. The statistical test alone is not enough -- on a low-variance
    series it fires constantly on rounding noise.
    """
    if len(series) < min_observations:
        return []

    ordered = sorted(series)
    floats = [(bucket, float(value)) for bucket, value in ordered]
    profile = seasonal_profile(floats)

    # Ratio of actual to seasonally expected. A flat series of 1.0s means "no surprises".
    ratios: list[float] = []
    for bucket, value in floats:
        expected = profile.get(_profile_key(bucket), 0.0)
        ratios.append(value / expected if expected > 0 else (1.0 if value == 0 else 2.0))

    scores = robust_z_scores(ratios)

    anomalies: list[Anomaly] = []
    for (bucket, value), score in zip(floats, scores, strict=True):
        baseline = Decimal(str(profile.get(_profile_key(bucket), 0.0)))
        candidate = _make(bucket, value, baseline, score)
        if (
            score > threshold
            and candidate.deviation_usd >= min_deviation_usd
            and candidate.multiple >= min_multiple
        ):
            anomalies.append(candidate)

    return sorted(anomalies, key=lambda a: a.bucket)


def _make(bucket: datetime, value: float, baseline: Decimal, score: float) -> Anomaly:
    observed = Decimal(str(value))
    return Anomaly(
        bucket=bucket,
        observed_usd=observed,
        baseline_usd=baseline,
        deviation_usd=observed - baseline,
        score=round(score, 4),
    )


def attribute_cause(
    observed: dict[tuple[str, str], Decimal],
    baseline: dict[tuple[str, str], Decimal],
    *,
    top_n: int = 3,
) -> tuple[Cause, ...]:
    """Explain a spike by the dimension values that grew most.

    ``observed`` and ``baseline`` map ``(dimension, value)`` to spend. The largest
    positive deltas are the probable cause, reported with their share of total growth.
    """
    deltas: list[tuple[tuple[str, str], Decimal]] = []
    for key in set(observed) | set(baseline):
        delta = observed.get(key, Decimal(0)) - baseline.get(key, Decimal(0))
        if delta > 0:
            deltas.append((key, delta))

    if not deltas:
        return ()

    total = sum(delta for _, delta in deltas)
    deltas.sort(key=lambda item: item[1], reverse=True)

    return tuple(
        Cause(
            dimension=dimension,
            value=value,
            delta_usd=delta,
            share=float(delta / total) if total else 0.0,
        )
        for (dimension, value), delta in deltas[:top_n]
    )


# --- database-backed detection -----------------------------------------------------

#: Dimensions searched for a probable cause, in the order a human would check them.
CAUSE_DIMENSIONS: tuple[str, ...] = ("model_key", "feature", "project")


def scan(
    conn: psycopg.Connection,
    *,
    filters: Filters,
    threshold: float = DEFAULT_THRESHOLD,
    dimensions: Sequence[str] = CAUSE_DIMENSIONS,
    min_deviation_usd: Decimal = DEFAULT_MIN_DEVIATION_USD,
) -> list[Anomaly]:
    """Detect anomalies over stored hourly spend and attribute each one to a cause."""
    series = queries.hourly_spend(conn, filters=filters)
    found = detect(series, threshold=threshold, min_deviation_usd=min_deviation_usd)
    if not found:
        return []

    # One query per dimension for the whole window, then attribute in Python. The
    # alternative -- two queries per anomaly -- costs more and buys nothing.
    tables: dict[str, dict[datetime, dict[str, Decimal]]] = {}
    for dimension in dimensions:
        table: dict[datetime, dict[str, Decimal]] = {}
        for row in queries.spend_series(
            conn, filters=filters, granularity="hour", group_by=[dimension]
        ):
            value = str(row[dimension] or "unknown")
            table.setdefault(row["bucket"], {})[value] = Decimal(row["cost_usd"])
        tables[dimension] = table

    return [replace(anomaly, probable_cause=_cause_for(anomaly, tables)) for anomaly in found]


def _cause_for(
    anomaly: Anomaly, tables: dict[str, dict[datetime, dict[str, Decimal]]]
) -> tuple[Cause, ...]:
    """Compare the spike hour against comparable hours, dimension by dimension."""
    season = _profile_key(anomaly.bucket)
    observed: dict[tuple[str, str], Decimal] = {}
    baseline: dict[tuple[str, str], Decimal] = {}

    for dimension, table in tables.items():
        for value, cost in table.get(anomaly.bucket, {}).items():
            observed[dimension, value] = cost

        samples: dict[str, list[Decimal]] = {}
        for bucket, values in table.items():
            # Comparable = same hour-of-day and day-type, and not the spike itself.
            if bucket == anomaly.bucket or _profile_key(bucket) != season:
                continue
            for value, cost in values.items():
                samples.setdefault(value, []).append(cost)
        for value, costs in samples.items():
            baseline[dimension, value] = statistics.median(costs)

    return attribute_cause(observed, baseline)


def persist(
    conn: psycopg.Connection, anomalies: Sequence[Anomaly], *, scope_key: str = "global"
) -> int:
    """Store anomalies idempotently: re-scanning a window updates rather than duplicates."""
    if not anomalies:
        return 0
    rows = [
        (
            anomaly.bucket,
            scope_key,
            anomaly.observed_usd,
            anomaly.baseline_usd,
            anomaly.deviation_usd,
            anomaly.score,
            Jsonb(
                [
                    {
                        "dimension": cause.dimension,
                        "value": cause.value,
                        "delta_usd": str(cause.delta_usd),
                        "share": round(cause.share, 6),
                    }
                    for cause in anomaly.probable_cause
                ]
            ),
        )
        for anomaly in anomalies
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO anomaly
                (bucket, scope_key, observed_usd, baseline_usd, deviation_usd, score,
                 probable_cause)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (bucket, scope_key) DO UPDATE SET
                observed_usd   = EXCLUDED.observed_usd,
                baseline_usd   = EXCLUDED.baseline_usd,
                deviation_usd  = EXCLUDED.deviation_usd,
                score          = EXCLUDED.score,
                probable_cause = EXCLUDED.probable_cause,
                detected_at    = now()
            """,
            rows,
        )
        return max(cur.rowcount, 0)


def recent(
    conn: psycopg.Connection, *, since: datetime, scope_key: str = "global", limit: int = 100
) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT bucket, scope_key, observed_usd, baseline_usd, deviation_usd, score, "
            "probable_cause, detected_at FROM anomaly "
            "WHERE bucket >= %s AND scope_key = %s ORDER BY bucket DESC LIMIT %s",
            (since, scope_key, limit),
        )
        return list(cur.fetchall())
