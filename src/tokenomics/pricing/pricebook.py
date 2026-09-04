"""Immutable, content-addressed pricing snapshots.

A :class:`PriceBook` is a frozen view of a model price list. Snapshots are addressed by
the SHA-256 of their source bytes, so a historical cost can always be recomputed from
exactly the numbers that produced it -- prices changing upstream never rewrites history.

A snapshot is vendored into the package, so Tokenomics runs **fully offline**. Refreshing
is an explicit, opt-in network call (``tokenomics pricing refresh``).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from tokenomics.models import Component, ServiceTier
from tokenomics.pricing.grammar import (
    CACHE_TTL_DEFAULT,
    RateKey,
    parse_rate_key,
    to_decimal,
)

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pricebook"
_VENDORED_GZ = _DATA_DIR / "litellm.json.gz"
_VENDORED_META = _DATA_DIR / "litellm.meta.json"


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Every rate known for one price-list key."""

    key: str
    provider: str | None
    mode: str | None
    max_input_tokens: int | None
    deprecation_date: str | None
    rates: dict[RateKey, Decimal]
    context_tiers: tuple[int, ...]

    def select_context_tier(self, input_tokens: int) -> int | None:
        """Threshold-select: the highest tier the request *exceeds*.

        Long-context pricing re-prices the whole request once the threshold is passed;
        it is not a graduated bracket. ``tiered_pricing`` ranges are normalized into the
        same thresholds at load time, because providers apply them the same way.
        """
        selected: int | None = None
        for tier in self.context_tiers:
            if input_tokens > tier:
                selected = tier
        return selected

    def rate(
        self,
        component: Component,
        *,
        context_tier: int | None,
        service_tier: ServiceTier = ServiceTier.STANDARD,
        cache_ttl: str = CACHE_TTL_DEFAULT,
    ) -> Decimal | None:
        """Look up a rate, degrading gracefully to less specific variants."""
        candidates = [
            RateKey(component, context_tier, service_tier, cache_ttl),
            RateKey(component, context_tier, ServiceTier.STANDARD, cache_ttl),
            RateKey(component, context_tier, service_tier, CACHE_TTL_DEFAULT),
            RateKey(component, context_tier, ServiceTier.STANDARD, CACHE_TTL_DEFAULT),
        ]
        if context_tier is not None:
            candidates += [
                RateKey(component, None, service_tier, cache_ttl),
                RateKey(component, None, ServiceTier.STANDARD, cache_ttl),
                RateKey(component, None, service_tier, CACHE_TTL_DEFAULT),
                RateKey(component, None, ServiceTier.STANDARD, CACHE_TTL_DEFAULT),
            ]
        for candidate in candidates:
            if (found := self.rates.get(candidate)) is not None:
                return found
        return None

    def prices(self, component: Component) -> bool:
        """Whether this key quotes a rate for a component at *any* tier.

        Distinct from :meth:`rate` returning ``None``: that answers "is there a rate
        for this exact request", whereas the what-if simulator needs to know whether
        the model prices caching at all before claiming a cache saving.
        """
        return any(k.component is component for k in self.rates)

    @property
    def is_priceable(self) -> bool:
        return self.prices(Component.INPUT) or self.prices(Component.OUTPUT)


def _build_model(key: str, raw: dict[str, Any]) -> ModelPricing:
    rates: dict[RateKey, Decimal] = {}
    tiers: set[int] = set()

    for field, value in raw.items():
        if "cost" not in field:
            continue
        rate_key = parse_rate_key(field)
        if rate_key is None:
            continue
        amount = to_decimal(value)
        if amount is None:
            continue
        rates[rate_key] = amount
        if rate_key.context_tier is not None:
            tiers.add(rate_key.context_tier)

    # `tiered_pricing` expresses the same threshold-select behaviour as explicit ranges.
    tiered = raw.get("tiered_pricing")
    if isinstance(tiered, list):
        for band in tiered:
            if not isinstance(band, dict):
                continue
            band_range = band.get("range") or [0, 0]
            lower = int(band_range[0]) if band_range else 0
            tier: int | None = lower if lower > 0 else None
            for field, value in band.items():
                if field == "range":
                    continue
                parsed = parse_rate_key(field)
                amount = to_decimal(value)
                if parsed is None or amount is None:
                    continue
                rates.setdefault(
                    RateKey(parsed.component, tier, parsed.service_tier, parsed.cache_ttl),
                    amount,
                )
            if tier is not None:
                tiers.add(tier)

    max_input = raw.get("max_input_tokens")
    return ModelPricing(
        key=key,
        provider=raw.get("litellm_provider"),
        mode=raw.get("mode"),
        max_input_tokens=int(max_input) if isinstance(max_input, int | float) else None,
        deprecation_date=raw.get("deprecation_date"),
        rates=rates,
        context_tiers=tuple(sorted(tiers)),
    )


@dataclass(frozen=True)
class PriceBook:
    """An immutable snapshot of a model price list."""

    snapshot_id: str
    source: str
    fetched_at: datetime
    models: dict[str, ModelPricing]

    def __len__(self) -> int:
        return len(self.models)

    def get(self, key: str) -> ModelPricing | None:
        return self.models.get(key)

    @classmethod
    def from_bytes(cls, payload: bytes, *, source: str, fetched_at: datetime) -> PriceBook:
        snapshot_id = hashlib.sha256(payload).hexdigest()
        raw = json.loads(payload)
        models = {
            key: _build_model(key, value)
            for key, value in raw.items()
            if isinstance(value, dict) and key != "sample_spec"
        }
        return cls(
            snapshot_id=snapshot_id,
            source=source,
            fetched_at=fetched_at,
            models=models,
        )

    @classmethod
    def vendored(cls) -> PriceBook:
        """Load the snapshot shipped with the package. Never touches the network."""
        payload = gzip.decompress(_VENDORED_GZ.read_bytes())
        meta = json.loads(_VENDORED_META.read_text())
        book = cls.from_bytes(
            payload,
            source=meta["source"],
            fetched_at=datetime.fromisoformat(meta["fetched_at"].replace("Z", "+00:00")),
        )
        if book.snapshot_id != meta["sha256"]:
            msg = (
                f"vendored pricebook integrity check failed: "
                f"expected {meta['sha256']}, got {book.snapshot_id}"
            )
            raise ValueError(msg)
        return book

    @classmethod
    def fetch(cls, url: str = LITELLM_URL, *, timeout: float = 30.0) -> PriceBook:
        """Download a fresh snapshot. The only outbound network call in the product."""
        import httpx

        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        return cls.from_bytes(response.content, source=url, fetched_at=datetime.now(UTC))
