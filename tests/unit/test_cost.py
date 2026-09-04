"""Cost computation tests.

Golden values are hand-computed from the frozen (content-addressed) snapshot pinned in
``conftest.FROZEN_SNAPSHOT_SHA``. If that hash changes, these must be re-derived
deliberately -- that is the point of pinning it.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tokenomics.models import ServiceTier, TokenVector
from tokenomics.pricing.cost import WARN_INCONSISTENT, compute_cost
from tokenomics.pricing.grammar import CACHE_TTL_1H


def price(book, key: str, tokens: TokenVector, **kwargs) -> Decimal:
    pricing = book.get(key)
    assert pricing is not None, f"{key} missing from snapshot"
    breakdown = compute_cost(tokens, pricing, model_key=key, snapshot_id="test", **kwargs)
    assert breakdown is not None
    return breakdown.total_usd


def test_snapshot_is_the_frozen_one(book, frozen_snapshot_sha):
    assert book.snapshot_id == frozen_snapshot_sha


# --------------------------------------------------------------------------- golden


def test_gpt4o_simple(book):
    # 1000 * 2.5e-6 + 500 * 1e-5
    assert price(book, "gpt-4o", TokenVector(input=1000, output=500)) == Decimal("0.0075")


def test_claude_below_long_context_threshold(book):
    # 1000 * 3e-6 + 500 * 1.5e-5
    assert price(book, "claude-sonnet-4-5", TokenVector(input=1000, output=500)) == Decimal(
        "0.0105"
    )


def test_claude_above_200k_reprices_the_whole_request(book):
    """Long-context pricing is threshold-select, not a graduated bracket."""
    # 250_000 * 6e-6 + 1000 * 2.25e-5   (both at the >200k tier)
    assert price(book, "claude-sonnet-4-5", TokenVector(input=250_000, output=1000)) == Decimal(
        "1.5225"
    )


def test_gemini_above_200k_tier(book):
    # 250_000 * 2.5e-6 + 1000 * 1.5e-5
    assert price(book, "gemini-2.5-pro", TokenVector(input=250_000, output=1000)) == Decimal(
        "0.640"
    )


def test_cache_read_partitions_out_of_input(book):
    """The headline correctness property: cached input is billed once, at the cache rate."""
    tokens = TokenVector(input=100_000, output=1000, cache_read=90_000)
    # billable input 10_000 * 3e-6 + cache_read 90_000 * 3e-7 + output 1000 * 1.5e-5
    assert price(book, "claude-sonnet-4-5", tokens) == Decimal("0.072")


def test_naive_summation_would_overstate(book):
    """Documents the bug this design exists to prevent."""
    tokens = TokenVector(input=100_000, output=1000, cache_read=90_000)
    correct = price(book, "claude-sonnet-4-5", tokens)
    naive = (
        Decimal(tokens.input) * Decimal("3e-6")
        + Decimal(tokens.output) * Decimal("1.5e-5")
        + Decimal(tokens.cache_read) * Decimal("3e-7")
    )
    assert naive > correct * Decimal("4")


def test_cache_write_uses_five_minute_rate_by_default(book):
    tokens = TokenVector(input=10_000, output=0, cache_write=10_000)
    # 10_000 * 1.25e-6, the 5-minute cache-write rate
    assert price(book, "claude-haiku-4-5-20251001", tokens) == Decimal("0.0125")


def test_cache_write_one_hour_rate_is_higher(book):
    tokens = TokenVector(input=10_000, output=0, cache_write=10_000)
    # 10_000 * 2e-6, the 1-hour cache-write rate
    assert price(book, "claude-haiku-4-5-20251001", tokens, cache_ttl=CACHE_TTL_1H) == Decimal(
        "0.02"
    )


def test_reasoning_billed_apart_when_the_model_prices_it_apart(book):
    """qwen-turbo: in 5e-8, out 2e-7, reasoning 5e-7."""
    tokens = TokenVector(input=1000, output=1000, reasoning=400)
    # 1000*5e-8 + (1000-400)*2e-7 + 400*5e-7
    assert price(book, "dashscope/qwen-turbo", tokens) == Decimal("0.00037")


def test_reasoning_is_not_double_billed_when_bundled(book):
    """Claude has no separate reasoning rate, so reasoning is already inside output."""
    without = price(book, "claude-sonnet-4-5", TokenVector(input=100, output=1000))
    with_reasoning = price(
        book, "claude-sonnet-4-5", TokenVector(input=100, output=1000, reasoning=800)
    )
    assert without == with_reasoning


def test_batch_service_tier_is_cheaper(book):
    standard = price(book, "gpt-4o", TokenVector(input=1000, output=500))
    batch = price(
        book, "gpt-4o", TokenVector(input=1000, output=500), service_tier=ServiceTier.BATCH
    )
    assert batch < standard


def test_unknown_service_tier_falls_back_to_standard(book):
    """A model with no flex pricing must still price, not return None."""
    standard = price(book, "claude-sonnet-4-5", TokenVector(input=1000, output=500))
    flex = price(
        book,
        "claude-sonnet-4-5",
        TokenVector(input=1000, output=500),
        service_tier=ServiceTier.FLEX,
    )
    assert flex == standard


# ------------------------------------------------------------------- inconsistency


def test_inconsistent_instrumentation_is_clamped_and_flagged(book):
    """Components exceeding their total must never yield a negative bill."""
    tokens = TokenVector(input=1000, output=10, cache_read=900, cache_write=900)
    pricing = book.get("claude-sonnet-4-5")
    breakdown = compute_cost(tokens, pricing, model_key="k", snapshot_id="s")
    assert breakdown is not None
    assert WARN_INCONSISTENT in breakdown.warnings
    assert breakdown.total_usd > 0
    assert breakdown.input_usd >= 0


def test_reasoning_exceeding_output_is_clamped(book):
    tokens = TokenVector(input=10, output=100, reasoning=500)
    pricing = book.get("dashscope/qwen-turbo")
    breakdown = compute_cost(tokens, pricing, model_key="k", snapshot_id="s")
    assert breakdown is not None
    assert WARN_INCONSISTENT in breakdown.warnings
    assert breakdown.output_usd >= 0


def test_model_without_token_rates_returns_none(book):
    """An image-only model resolves but cannot be token-priced."""
    from tokenomics.pricing.pricebook import ModelPricing

    empty = ModelPricing(
        key="image-only",
        provider="x",
        mode="image_generation",
        max_input_tokens=None,
        deprecation_date=None,
        rates={},
        context_tiers=(),
    )
    result = compute_cost(TokenVector(input=1, output=1), empty, model_key="k", snapshot_id="s")
    assert result is None


# ------------------------------------------------------------------------ properties

_counts = st.integers(min_value=0, max_value=2_000_000)


@st.composite
def consistent_vectors(draw) -> TokenVector:
    """Token vectors that satisfy the semconv inclusion invariants."""
    input_tokens = draw(_counts)
    output_tokens = draw(_counts)
    cache_read = draw(st.integers(min_value=0, max_value=input_tokens))
    cache_write = draw(st.integers(min_value=0, max_value=input_tokens - cache_read))
    reasoning = draw(st.integers(min_value=0, max_value=output_tokens))
    return TokenVector(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        reasoning=reasoning,
    )


@settings(max_examples=200, deadline=None)
@given(tokens=consistent_vectors())
@pytest.mark.parametrize("key", ["claude-sonnet-4-5", "gpt-4o", "dashscope/qwen-turbo"])
def test_cost_is_never_negative(book, tokens, key):
    breakdown = compute_cost(tokens, book.get(key), model_key=key, snapshot_id="s")
    assert breakdown is not None
    assert breakdown.total_usd >= 0
    assert all(v >= 0 for v in breakdown.billable_units.values())


@settings(max_examples=200, deadline=None)
@given(tokens=consistent_vectors())
def test_components_sum_to_total(book, tokens):
    breakdown = compute_cost(tokens, book.get("claude-sonnet-4-5"), model_key="k", snapshot_id="s")
    assert breakdown is not None
    assert sum(breakdown.billable_units.values()) == breakdown.total_usd


@settings(max_examples=100, deadline=None)
@given(
    input_tokens=st.integers(min_value=1, max_value=150_000),
    extra=st.integers(min_value=1, max_value=10_000),
)
def test_cost_is_monotonic_in_input(book, input_tokens, extra):
    pricing = book.get("claude-sonnet-4-5")
    smaller = compute_cost(
        TokenVector(input=input_tokens, output=10), pricing, model_key="k", snapshot_id="s"
    )
    larger = compute_cost(
        TokenVector(input=input_tokens + extra, output=10), pricing, model_key="k", snapshot_id="s"
    )
    assert smaller is not None and larger is not None
    assert larger.total_usd >= smaller.total_usd


@settings(max_examples=100, deadline=None)
@given(
    input_tokens=st.integers(min_value=100, max_value=150_000),
    output_tokens=st.integers(min_value=0, max_value=5_000),
)
def test_caching_never_costs_more_than_not_caching(book, input_tokens, output_tokens):
    """A cache read is a discount; it must never increase the bill."""
    pricing = book.get("claude-sonnet-4-5")
    uncached = compute_cost(
        TokenVector(input=input_tokens, output=output_tokens),
        pricing,
        model_key="k",
        snapshot_id="s",
    )
    cached = compute_cost(
        TokenVector(input=input_tokens, output=output_tokens, cache_read=input_tokens),
        pricing,
        model_key="k",
        snapshot_id="s",
    )
    assert uncached is not None and cached is not None
    assert cached.total_usd <= uncached.total_usd
