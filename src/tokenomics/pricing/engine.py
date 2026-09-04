"""The pricing engine: resolve a model, then price a call against it."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from tokenomics.models import CostBreakdown, ServiceTier, TokenVector, UsageEvent
from tokenomics.pricing.cost import compute_cost
from tokenomics.pricing.pricebook import PriceBook
from tokenomics.pricing.resolver import Method, ModelResolver, Resolution


@dataclass(frozen=True, slots=True)
class PricedEvent:
    """An event plus the audit trail of how (or whether) it was priced."""

    event: UsageEvent
    resolution: Resolution

    @property
    def is_priced(self) -> bool:
        return self.event.cost is not None


class PricingEngine:
    """Prices usage against a single immutable snapshot."""

    def __init__(self, book: PriceBook) -> None:
        self.book = book
        self.resolver = ModelResolver(book)

    @property
    def snapshot_id(self) -> str:
        return self.book.snapshot_id

    def price_tokens(
        self,
        tokens: TokenVector,
        *,
        model: str | None,
        provider: str | None = None,
        fallback_model: str | None = None,
        service_tier: ServiceTier = ServiceTier.STANDARD,
    ) -> tuple[CostBreakdown | None, Resolution]:
        """Price a raw token vector. Used by ingestion and by the what-if simulator."""
        resolution = self.resolver.resolve(model, provider=provider, fallback=fallback_model)
        if resolution.pricing is None or resolution.model_key is None:
            return None, resolution

        breakdown = compute_cost(
            tokens,
            resolution.pricing,
            model_key=resolution.model_key,
            snapshot_id=self.book.snapshot_id,
            service_tier=service_tier,
        )
        if breakdown is None:
            # The key resolved but carries no token rates (e.g. an image-only model).
            return None, Resolution(Method.UNPRICED, candidates=resolution.candidates)
        return breakdown, resolution

    def price(self, event: UsageEvent) -> PricedEvent:
        """Attach cost to an event. Unpriceable events keep ``cost=None``, never zero."""
        breakdown, resolution = self.price_tokens(
            event.tokens,
            model=event.response_model or event.request_model,
            provider=event.provider,
            fallback_model=event.request_model,
            service_tier=event.service_tier,
        )
        return PricedEvent(
            event=event.model_copy(update={"cost": breakdown}),
            resolution=resolution,
        )


@lru_cache(maxsize=1)
def default_engine() -> PricingEngine:
    """The engine backed by the vendored snapshot. Loaded once, never hits the network."""
    return PricingEngine(PriceBook.vendored())
