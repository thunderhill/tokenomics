from __future__ import annotations

import pytest

from tokenomics.models import Component, ServiceTier
from tokenomics.pricing.grammar import (
    CACHE_TTL_1H,
    CACHE_TTL_DEFAULT,
    UNPRICED_TOKEN_FIELDS,
    parse_rate_key,
)


@pytest.mark.parametrize(
    ("field", "component", "tier", "service", "ttl"),
    [
        ("input_cost_per_token", Component.INPUT, None, ServiceTier.STANDARD, CACHE_TTL_DEFAULT),
        ("output_cost_per_token", Component.OUTPUT, None, ServiceTier.STANDARD, CACHE_TTL_DEFAULT),
        (
            "output_cost_per_reasoning_token",
            Component.REASONING,
            None,
            ServiceTier.STANDARD,
            CACHE_TTL_DEFAULT,
        ),
        (
            "cache_read_input_token_cost",
            Component.CACHE_READ,
            None,
            ServiceTier.STANDARD,
            CACHE_TTL_DEFAULT,
        ),
        (
            "cache_creation_input_token_cost",
            Component.CACHE_WRITE,
            None,
            ServiceTier.STANDARD,
            CACHE_TTL_DEFAULT,
        ),
        (
            "input_cost_per_token_cache_hit",
            Component.CACHE_READ,
            None,
            ServiceTier.STANDARD,
            CACHE_TTL_DEFAULT,
        ),
        (
            "input_cost_per_token_above_200k_tokens",
            Component.INPUT,
            200_000,
            ServiceTier.STANDARD,
            CACHE_TTL_DEFAULT,
        ),
        (
            "input_cost_per_token_above_128k_tokens",
            Component.INPUT,
            128_000,
            ServiceTier.STANDARD,
            CACHE_TTL_DEFAULT,
        ),
        (
            "input_cost_per_token_batches",
            Component.INPUT,
            None,
            ServiceTier.BATCH,
            CACHE_TTL_DEFAULT,
        ),
        (
            "input_cost_per_token_priority",
            Component.INPUT,
            None,
            ServiceTier.PRIORITY,
            CACHE_TTL_DEFAULT,
        ),
        (
            "output_cost_per_token_above_272k_tokens_flex",
            Component.OUTPUT,
            272_000,
            ServiceTier.FLEX,
            CACHE_TTL_DEFAULT,
        ),
        (
            "cache_read_input_token_cost_above_200k_tokens_priority",
            Component.CACHE_READ,
            200_000,
            ServiceTier.PRIORITY,
            CACHE_TTL_DEFAULT,
        ),
        (
            "cache_creation_input_token_cost_above_1hr",
            Component.CACHE_WRITE,
            None,
            ServiceTier.STANDARD,
            CACHE_TTL_1H,
        ),
        (
            "cache_creation_input_token_cost_above_1hr_above_200k_tokens",
            Component.CACHE_WRITE,
            200_000,
            ServiceTier.STANDARD,
            CACHE_TTL_1H,
        ),
    ],
)
def test_parses_every_real_spelling(field, component, tier, service, ttl):
    key = parse_rate_key(field)
    assert key is not None
    assert (key.component, key.context_tier, key.service_tier, key.cache_ttl) == (
        component,
        tier,
        service,
        ttl,
    )


@pytest.mark.parametrize(
    "field",
    [
        "input_cost_per_image",
        "output_cost_per_second",
        "input_cost_per_character",
        "search_context_cost_per_query",
        "input_cost_per_audio_token",
        "input_dbu_cost_per_token",
        "supports_vision",
        "litellm_provider",
        "mode",
    ],
)
def test_ignores_non_text_token_units(field):
    """v0.1 prices text tokens only; other units must be skipped, not misparsed."""
    assert parse_rate_key(field) is None


def test_grammar_covers_every_token_cost_field_in_the_book():
    """Any unparsed *token* cost field would silently understate cost.

    Guards the parser against upstream adding a new spelling we would drop on the floor.
    """
    import gzip
    import json
    from pathlib import Path

    import tokenomics

    data = Path(tokenomics.__file__).parent / "data" / "pricebook" / "litellm.json.gz"
    raw = json.loads(gzip.decompress(data.read_bytes()))

    # Units this version does not price. Text tokens are the only priced unit in v0.1.
    non_token_units = (
        "image",
        "audio",
        "video",
        "character",
        "pixel",
        "second",
        "query",
        "page",
        "session",
        "credit",
        "dbu",
        "request",
        "gb",
        "calls",
        "1k_tokens",
        "unit",
    )
    unparsed = {
        field
        for entry in raw.values()
        if isinstance(entry, dict)
        for field in entry
        if "cost" in field
        and parse_rate_key(field) is None
        and not any(unit in field for unit in non_token_units)
        and field not in UNPRICED_TOKEN_FIELDS
    }
    assert unparsed == set(), f"unhandled token cost spellings: {sorted(unparsed)}"
