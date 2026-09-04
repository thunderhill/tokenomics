"""What-if: reprice historical traffic against a different model.

The question this answers is "what would last month have cost on model X?" -- and the
honest answer has two traps in it.

**Cache pricing does not transfer.** A workload tuned for Anthropic prompt caching can
have 80% of its input tokens served at 0.1x. Reprice it against a model with no cache
rates and those tokens fall back to the full input rate. The cost function already
does that fold-back correctly, but a simulation that does not *say so* looks like the
target model is far more expensive than it needs to be, when the real finding is
"you would need to re-engineer the prompt". So every simulation reports how many cached
tokens got re-priced at full rate.

**Long-context tiers are per request.** Threshold-select pricing reprices the whole
request once it exceeds the threshold, so aggregate token totals cannot be repriced --
a million tokens spread over 1000 small requests costs differently from the same
million in 5 huge ones. Simulation therefore replays *token vectors*, not totals.

Quality is explicitly **not** modelled. The fields exist, are labelled ``not-measured``,
and are yours to fill from your own evals. A cost tool that guessed at quality would be
worse than one that admits it does not know.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from tokenomics.models import Component, ServiceTier, TokenVector
from tokenomics.pricing.engine import PricingEngine
from tokenomics.pricing.pricebook import ModelPricing
from tokenomics.storage.queries import Filters

#: Distinct token-shapes to replay before falling back to sampling. Requests repeat a
#: lot in practice (same system prompt, similar answers), so this collapses hard.
DEFAULT_MAX_SHAPES = 20_000

WARN_CACHE_NOT_PRICED = "target_model_has_no_cache_pricing"
WARN_UNPRICED_BASELINE = "baseline_contains_unpriced_events"
WARN_SAMPLED = "sampled"
WARN_CONTEXT_OVERFLOW = "requests_exceed_target_context_window"


@dataclass(frozen=True, slots=True)
class Sample:
    """One token shape and how many times it occurred."""

    tokens: TokenVector
    requests: int = 1
    service_tier: ServiceTier = ServiceTier.STANDARD
    baseline_usd: Decimal | None = None  # None = the original event was unpriced


@dataclass(frozen=True, slots=True)
class QualityAssumption:
    """Placeholders. Tokenomics measures money, not answer quality.

    Fill these from your own evaluation suite; nothing here is inferred, and no default
    implies that a cheaper model is equivalent.
    """

    status: str = "not-measured"
    evaluation_suite: str | None = None
    accepted_quality_delta: float | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class Simulation:
    """The result of repricing a body of traffic against one target model."""

    target_model: str
    resolved_model_key: str | None
    snapshot_id: str
    requests: int
    repriced_requests: int
    unpriceable_requests: int
    baseline_usd: Decimal
    projected_usd: Decimal
    baseline_unpriced_requests: int = 0
    refolded_cache_tokens: int = 0
    scale_factor: float = 1.0
    warnings: tuple[str, ...] = ()
    quality: QualityAssumption = field(default_factory=QualityAssumption)

    @property
    def delta_usd(self) -> Decimal:
        return self.projected_usd - self.baseline_usd

    @property
    def delta_pct(self) -> float | None:
        if self.baseline_usd == 0:
            return None
        return float(self.delta_usd / self.baseline_usd * 100)

    @property
    def is_complete(self) -> bool:
        """Whether the comparison covers all the traffic asked about."""
        return (
            self.unpriceable_requests == 0
            and self.baseline_unpriced_requests == 0
            and self.scale_factor == 1.0
        )


def simulate(
    engine: PricingEngine,
    samples: Sequence[Sample],
    *,
    target_model: str,
    provider: str | None = None,
    keep_service_tier: bool = True,
    scale_factor: float = 1.0,
    quality: QualityAssumption | None = None,
) -> Simulation:
    """Reprice ``samples`` against ``target_model``."""
    resolution = engine.resolver.resolve(target_model, provider=provider)
    pricing: ModelPricing | None = resolution.pricing

    requests = sum(s.requests for s in samples)
    baseline = sum((s.baseline_usd or Decimal(0)) * s.requests for s in samples)
    baseline_unpriced = sum(s.requests for s in samples if s.baseline_usd is None)

    warnings: set[str] = set()
    if baseline_unpriced:
        warnings.add(WARN_UNPRICED_BASELINE)
    if scale_factor != 1.0:
        warnings.add(WARN_SAMPLED)

    if pricing is None or resolution.model_key is None:
        return Simulation(
            target_model=target_model,
            resolved_model_key=None,
            snapshot_id=engine.snapshot_id,
            requests=requests,
            repriced_requests=0,
            unpriceable_requests=requests,
            baseline_usd=_scaled(Decimal(baseline), scale_factor),
            projected_usd=Decimal(0),
            baseline_unpriced_requests=baseline_unpriced,
            scale_factor=scale_factor,
            warnings=tuple(sorted(warnings)),
            quality=quality or QualityAssumption(),
        )

    prices_cache = pricing.prices(Component.CACHE_READ)
    window = pricing.max_input_tokens

    projected = Decimal(0)
    repriced = 0
    unpriceable = 0
    refolded = 0

    for sample in samples:
        tier = sample.service_tier if keep_service_tier else ServiceTier.STANDARD
        breakdown, _ = engine.price_tokens(
            sample.tokens,
            model=resolution.model_key,
            service_tier=tier,
        )
        if breakdown is None:
            unpriceable += sample.requests
            continue
        projected += breakdown.total_usd * sample.requests
        repriced += sample.requests
        if not prices_cache:
            # compute_cost already billed these at the input rate; count them so the
            # report can say *why* the target looks expensive.
            refolded += (sample.tokens.cache_read + sample.tokens.cache_write) * sample.requests
        if window is not None and sample.tokens.input > window:
            warnings.add(WARN_CONTEXT_OVERFLOW)

    if refolded:
        warnings.add(WARN_CACHE_NOT_PRICED)
    if unpriceable:
        warnings.add("some_requests_unpriceable_on_target")

    return Simulation(
        target_model=target_model,
        resolved_model_key=resolution.model_key,
        snapshot_id=engine.snapshot_id,
        requests=requests,
        repriced_requests=repriced,
        unpriceable_requests=unpriceable,
        baseline_usd=_scaled(Decimal(baseline), scale_factor),
        projected_usd=_scaled(projected, scale_factor),
        baseline_unpriced_requests=baseline_unpriced,
        refolded_cache_tokens=int(refolded * scale_factor),
        scale_factor=scale_factor,
        warnings=tuple(sorted(warnings)),
        quality=quality or QualityAssumption(),
    )


def compare(
    engine: PricingEngine,
    samples: Sequence[Sample],
    *,
    targets: Iterable[str],
    **kwargs: Any,
) -> list[Simulation]:
    """Reprice the same traffic against several models, cheapest projection first."""
    results = [simulate(engine, samples, target_model=t, **kwargs) for t in targets]
    return sorted(results, key=lambda s: (s.resolved_model_key is None, s.projected_usd))


def _scaled(amount: Decimal, factor: float) -> Decimal:
    return amount if factor == 1.0 else amount * Decimal(str(factor))


def load_samples(
    conn: psycopg.Connection,
    *,
    filters: Filters,
    max_shapes: int = DEFAULT_MAX_SHAPES,
) -> tuple[list[Sample], float]:
    """Collapse stored events into distinct token shapes.

    Returns ``(samples, scale_factor)``. Identical requests collapse into one row with
    a count, which keeps the replay exact. If the traffic is more varied than
    ``max_shapes``, we take a uniform random subset and return the factor needed to
    scale the result back up -- reported as a ``sampled`` warning rather than passed
    off as exact.
    """
    where, params = filters.where()
    params["limit"] = max_shapes + 1

    query = sql.SQL(
        """
        SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
               reasoning_tokens, service_tier,
               count(*) AS requests,
               sum(cost_usd) AS cost_usd,
               count(*) FILTER (WHERE cost_usd IS NULL) AS unpriced
        FROM usage_event
        WHERE {where}
        GROUP BY 1, 2, 3, 4, 5, 6
        ORDER BY random()
        LIMIT %(limit)s
        """
    ).format(where=where)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        rows = list(cur.fetchall())

    truncated = len(rows) > max_shapes
    if truncated:
        rows = rows[:max_shapes]

    samples = [
        Sample(
            tokens=TokenVector(
                input=row["input_tokens"],
                output=row["output_tokens"],
                cache_read=row["cache_read_tokens"],
                cache_write=row["cache_write_tokens"],
                reasoning=row["reasoning_tokens"],
            ),
            requests=int(row["requests"]),
            service_tier=ServiceTier(row["service_tier"]),
            # A shape is only a priced baseline if *every* event in it was priced.
            baseline_usd=(
                Decimal(row["cost_usd"]) / int(row["requests"])
                if row["unpriced"] == 0 and row["cost_usd"] is not None
                else None
            ),
        )
        for row in rows
    ]

    if not truncated:
        return samples, 1.0

    covered = sum(s.requests for s in samples)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT count(*) FROM usage_event WHERE {where}").format(where=where), params
        )
        row = cur.fetchone()
    total = int(row[0]) if row else covered
    return samples, (total / covered if covered else 1.0)
