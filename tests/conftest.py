from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tokenomics.models import Component, ServiceTier
from tokenomics.pricing.engine import PricingEngine
from tokenomics.pricing.grammar import CACHE_TTL_DEFAULT, RateKey
from tokenomics.pricing.pricebook import ModelPricing, PriceBook

#: The vendored snapshot is content-addressed, so golden values below are stable.
#: If this assertion ever fails, the pricebook was refreshed and every golden
#: value in the suite must be re-derived deliberately rather than silently drift.
FROZEN_SNAPSHOT_SHA = "0e8d23239c2768bd096249dae81c631f940fa99c81b5206af15a822e07cf7d3d"


@pytest.fixture(scope="session")
def frozen_snapshot_sha() -> str:
    return FROZEN_SNAPSHOT_SHA


@pytest.fixture(scope="session")
def book() -> PriceBook:
    return PriceBook.vendored()


@pytest.fixture(scope="session")
def engine(book: PriceBook) -> PricingEngine:
    return PricingEngine(book)


def make_pricing(**rates: str) -> ModelPricing:
    """Build a synthetic ModelPricing for edge cases the real book does not contain."""
    table: dict[RateKey, Decimal] = {}
    tiers: set[int] = set()
    for name, value in rates.items():
        component_name, _, tier_text = name.partition("__")
        tier = int(tier_text) if tier_text else None
        table[RateKey(Component(component_name), tier, ServiceTier.STANDARD, CACHE_TTL_DEFAULT)] = (
            Decimal(value)
        )
        if tier is not None:
            tiers.add(tier)
    return ModelPricing(
        key="synthetic",
        provider="test",
        mode="chat",
        max_input_tokens=None,
        deprecation_date=None,
        rates=table,
        context_tiers=tuple(sorted(tiers)),
    )


@pytest.fixture
def synthetic_book() -> PriceBook:
    return PriceBook(
        snapshot_id="synthetic",
        source="test",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        models={"synthetic": make_pricing(input="0.000001", output="0.000002")},
    )
