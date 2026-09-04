"""Parser for the LiteLLM price-list key grammar.

The price list does not use flat fields. Cost keys are *compositional*, built from a
component, a unit, and optional modifiers::

    input_cost_per_token
    input_cost_per_token_above_200k_tokens
    cache_read_input_token_cost_above_200k_tokens_priority
    cache_creation_input_token_cost_above_1hr_above_200k_tokens
    output_cost_per_reasoning_token

Across 3111 models that yields ~40 distinct spellings in two naming families:

* ``{io}_cost_per_{unit}``          — e.g. ``input_cost_per_token``
* ``{component}_input_token_cost``  — e.g. ``cache_read_input_token_cost``

Parsing the grammar (rather than hard-coding field names) means new provider fields are
absorbed automatically instead of being silently dropped, which would understate cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from tokenomics.models import Component, ServiceTier

_CTX_TIER = re.compile(r"_above_(\d+)k_tokens$")

_SERVICE_TIERS: dict[str, ServiceTier] = {
    "_priority": ServiceTier.PRIORITY,
    "_flex": ServiceTier.FLEX,
    "_batches": ServiceTier.BATCH,
}

# Base spellings, after all modifier suffixes have been stripped.
_BASES: dict[str, Component] = {
    "input_cost_per_token": Component.INPUT,
    "output_cost_per_token": Component.OUTPUT,
    "output_cost_per_reasoning_token": Component.REASONING,
    "cache_read_input_token_cost": Component.CACHE_READ,
    "cache_creation_input_token_cost": Component.CACHE_WRITE,
    # DeepSeek-style spelling for a cache hit.
    "input_cost_per_token_cache_hit": Component.CACHE_READ,
}

#: Token-denominated cost fields that v0.1 deliberately does not price.
#:
#: These are real per-token costs, but each applies to a single model and has no
#: corresponding OpenTelemetry GenAI attribute to source a count from, so pricing them
#: would be guesswork. Listed explicitly (rather than filtered by a loose substring)
#: so the omission stays visible and a future version can pick them up.
UNPRICED_TOKEN_FIELDS = frozenset(
    {
        "citation_cost_per_token",  # perplexity/sonar-deep-research
    }
)

#: Long-cache TTL marker. Anthropic prices a 1-hour cache write above the 5-minute one.
CACHE_TTL_1H = "1h"
CACHE_TTL_DEFAULT = "5m"


@dataclass(frozen=True, slots=True)
class RateKey:
    """A fully-qualified coordinate into a model's rate table."""

    component: Component
    context_tier: int | None = None
    service_tier: ServiceTier = ServiceTier.STANDARD
    cache_ttl: str = CACHE_TTL_DEFAULT


def parse_rate_key(field: str) -> RateKey | None:
    """Parse a price-list field name into a :class:`RateKey`.

    Returns ``None`` for fields that are not per-token costs (per-image, per-character,
    per-second, per-query, capability flags, metadata), which v0.1 does not price.
    """
    remaining = field
    service_tier = ServiceTier.STANDARD
    context_tier: int | None = None
    cache_ttl = CACHE_TTL_DEFAULT

    # Modifiers are stripped from the end, innermost last. Order matters:
    # `..._above_1hr_above_200k_tokens_priority` peels tier, then ctx, then ttl.
    changed = True
    while changed:
        changed = False

        for suffix, tier in _SERVICE_TIERS.items():
            if remaining.endswith(suffix):
                remaining = remaining[: -len(suffix)]
                service_tier = tier
                changed = True
                break
        if changed:
            continue

        if match := _CTX_TIER.search(remaining):
            context_tier = int(match.group(1)) * 1_000
            remaining = remaining[: match.start()]
            changed = True
            continue

        if remaining.endswith("_above_1hr"):
            remaining = remaining[: -len("_above_1hr")]
            cache_ttl = CACHE_TTL_1H
            changed = True

    if field in UNPRICED_TOKEN_FIELDS:
        return None

    component = _BASES.get(remaining)
    if component is None:
        return None
    return RateKey(
        component=component,
        context_tier=context_tier,
        service_tier=service_tier,
        cache_ttl=cache_ttl,
    )


def to_decimal(value: object) -> Decimal | None:
    """Convert a price-list number to ``Decimal`` without float round-tripping."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            return Decimal(value)
        except (ValueError, ArithmeticError):
            return None
    return None
