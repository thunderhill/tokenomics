"""Cost computation by **partition**.

This module exists because of one fact that is easy to miss and expensive to get wrong.

OpenTelemetry GenAI reports token counts *inclusively*::

    cache_read + cache_write <= input_tokens
    reasoning                <= output_tokens

Provider price lists quote *mutually exclusive* per-component rates: the input rate is
for input that was **not** served from cache, and the cache-read rate replaces it for
input that was. Summing the reported components therefore bills the cached tokens twice
-- once at the input rate and once at the cache rate. Since a cache read is typically
**0.1x** the input rate, that overstates a cache-heavy Anthropic workload by nearly
**10x** on its cached portion.

So we partition the reported totals into disjoint buckets before applying any rate::

    billable_input = input - cache_read - cache_write

The same applies on the output side: ``reasoning`` is already inside ``output_tokens``,
so it is only ever billed separately for the ~58 models that price it apart -- and then
the remainder, not the total, is billed at the output rate.
"""

from __future__ import annotations

from decimal import Decimal

from tokenomics.models import Component, CostBreakdown, ServiceTier, TokenVector
from tokenomics.pricing.grammar import CACHE_TTL_DEFAULT
from tokenomics.pricing.pricebook import ModelPricing

#: Money is stored and aggregated at this precision (matches NUMERIC(24,12) in Postgres).
QUANTUM = Decimal("0.000000000001")

WARN_INCONSISTENT = "partition_inconsistent"
WARN_MISSING_INPUT_RATE = "missing_rate:input"
WARN_MISSING_OUTPUT_RATE = "missing_rate:output"


def _partition_input(tokens: TokenVector) -> tuple[int, int, int, list[str]]:
    """Split reported input into (billable, cache_read, cache_write) disjoint buckets."""
    warnings: list[str] = []
    cache_read = tokens.cache_read
    cache_write = tokens.cache_write
    cached = cache_read + cache_write

    if cached > tokens.input:
        # Inconsistent instrumentation. Preserve the *total* (which the provider billed
        # us for) and scale the components down proportionally, rather than emitting a
        # negative billable figure or silently discarding the excess.
        warnings.append(WARN_INCONSISTENT)
        if cached > 0:
            scale = Decimal(tokens.input) / Decimal(cached)
            cache_read = int(Decimal(cache_read) * scale)
            cache_write = min(tokens.input - cache_read, int(Decimal(cache_write) * scale))

    billable = max(tokens.input - cache_read - cache_write, 0)
    return billable, cache_read, cache_write, warnings


def compute_cost(
    tokens: TokenVector,
    pricing: ModelPricing,
    *,
    model_key: str,
    snapshot_id: str,
    service_tier: ServiceTier = ServiceTier.STANDARD,
    cache_ttl: str = CACHE_TTL_DEFAULT,
) -> CostBreakdown | None:
    """Price one call. Returns ``None`` when the model has no usable token rates."""
    context_tier = pricing.select_context_tier(tokens.input)

    def rate_for(component: Component) -> Decimal | None:
        return pricing.rate(
            component,
            context_tier=context_tier,
            service_tier=service_tier,
            cache_ttl=cache_ttl,
        )

    input_rate = rate_for(Component.INPUT)
    output_rate = rate_for(Component.OUTPUT)
    if input_rate is None and output_rate is None:
        return None

    warnings: list[str] = []
    if input_rate is None:
        input_rate = Decimal(0)
        warnings.append(WARN_MISSING_INPUT_RATE)
    if output_rate is None:
        output_rate = Decimal(0)
        warnings.append(WARN_MISSING_OUTPUT_RATE)

    # A model with no separate cache rate bills cached tokens at the ordinary rate.
    cache_read_rate = rate_for(Component.CACHE_READ) or input_rate
    cache_write_rate = rate_for(Component.CACHE_WRITE) or input_rate
    reasoning_rate = rate_for(Component.REASONING)

    billable_input, cache_read, cache_write, partition_warnings = _partition_input(tokens)
    warnings.extend(partition_warnings)

    input_usd = Decimal(billable_input) * input_rate
    cache_read_usd = Decimal(cache_read) * cache_read_rate
    cache_write_usd = Decimal(cache_write) * cache_write_rate

    reasoning = tokens.reasoning
    if reasoning > tokens.output:
        warnings.append(WARN_INCONSISTENT)
        reasoning = tokens.output

    if reasoning_rate is not None:
        # Reasoning is priced apart, so bill the *remainder* at the output rate.
        output_usd = Decimal(tokens.output - reasoning) * output_rate
        reasoning_usd = Decimal(reasoning) * reasoning_rate
    else:
        # Reasoning is already inside output_tokens: billing it again would double-count.
        output_usd = Decimal(tokens.output) * output_rate
        reasoning_usd = Decimal(0)

    total = input_usd + output_usd + cache_read_usd + cache_write_usd + reasoning_usd

    rates_applied = {
        "input": input_rate,
        "output": output_rate,
        "cache_read": cache_read_rate,
        "cache_write": cache_write_rate,
    }
    if reasoning_rate is not None:
        rates_applied["reasoning"] = reasoning_rate

    return CostBreakdown(
        total_usd=total.quantize(QUANTUM),
        input_usd=input_usd.quantize(QUANTUM),
        output_usd=output_usd.quantize(QUANTUM),
        cache_read_usd=cache_read_usd.quantize(QUANTUM),
        cache_write_usd=cache_write_usd.quantize(QUANTUM),
        reasoning_usd=reasoning_usd.quantize(QUANTUM),
        model_key=model_key,
        snapshot_id=snapshot_id,
        context_tier=context_tier,
        service_tier=service_tier,
        rates_applied=rates_applied,
        warnings=tuple(dict.fromkeys(warnings)),
    )
