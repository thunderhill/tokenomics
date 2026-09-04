"""Model-key resolution.

Model identity is many-to-one. ``claude-sonnet-4-5`` appears under 20 distinct price-list
keys across Bedrock, Vertex, Azure, Databricks, Snowflake and regional prefixes, and 2535
of 3111 keys are namespaced. A plain ``dict[model]`` lookup misses constantly and returns
``0.0``, which is the worst possible failure for a cost tool: spend silently looks better
than it is.

So resolution is an explicit ladder with a recorded outcome. Every rung that fires is
stamped onto the event, and exhausting the ladder yields ``UNPRICED`` -- never zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

from tokenomics.pricing.pricebook import ModelPricing, PriceBook

# Regional / partition prefixes used by Bedrock inference profiles.
_REGION_PREFIXES = ("us-gov.", "us.", "eu.", "apac.", "au.", "jp.", "ca.", "global.")

# Vendor namespaces that appear as a dotted prefix on Bedrock model ids.
_VENDOR_PREFIXES = (
    "anthropic.",
    "amazon.",
    "meta.",
    "mistral.",
    "cohere.",
    "ai21.",
    "deepseek.",
    "qwen.",
    "writer.",
    "luma.",
    "stability.",
    "twelvelabs.",
)

_REPACKAGER_PREFIXES = ("databricks-", "accounts/fireworks/models/")

# Version / date stamps that do not change price.
_SUFFIX_PATTERNS = (
    re.compile(r"-v\d+:\d+$"),  # bedrock:  -v1:0
    re.compile(r":\d+$"),  # bedrock:  :0
    re.compile(r"@\d{8}$"),  # vertex:   @20250929
    re.compile(r"-\d{8}$"),  # openai:   -20250929
    re.compile(r"-\d{4}-\d{2}-\d{2}$"),  # openai:  -2024-08-06
)


class Method(StrEnum):
    """Which rung of the ladder produced the match. Stored for auditability."""

    EXACT = "exact"
    PROVIDER_QUALIFIED = "provider_qualified"
    NORMALIZED = "normalized"
    UNPRICED = "unpriced"


@dataclass(frozen=True, slots=True)
class Resolution:
    """The outcome of resolving a model name against a price book."""

    method: Method
    model_key: str | None = None
    pricing: ModelPricing | None = None
    candidates: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        return self.pricing is not None


def normalize(name: str) -> str:
    """Reduce a model name to a comparable base form.

    ``bedrock/us-gov-east-1/anthropic.claude-sonnet-4-5-20250929-v1:0`` -> ``claude-sonnet-4-5``
    """
    candidate = name.strip().lower()

    # Namespaced keys: the model id is the final path segment.
    if "/" in candidate:
        candidate = candidate.rsplit("/", 1)[-1]

    for prefix in _REGION_PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    for prefix in _VENDOR_PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    for prefix in _REPACKAGER_PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break

    changed = True
    while changed:
        changed = False
        for pattern in _SUFFIX_PATTERNS:
            if pattern.search(candidate):
                candidate = pattern.sub("", candidate)
                changed = True
    return candidate


class ModelResolver:
    """Resolves model names to :class:`ModelPricing` against one price book."""

    def __init__(self, book: PriceBook) -> None:
        self.book = book

    @cached_property
    def _normalized_index(self) -> dict[str, tuple[str, ...]]:
        """normalized name -> candidate keys, best first."""
        index: dict[str, list[str]] = {}
        for key in self.book.models:
            index.setdefault(normalize(key), []).append(key)
        # Prefer the plainest key: fewest namespace segments, then shortest, then stable.
        return {
            base: tuple(sorted(keys, key=lambda k: (k.count("/"), k.count("."), len(k), k)))
            for base, keys in index.items()
        }

    def resolve(
        self,
        model: str | None,
        *,
        provider: str | None = None,
        fallback: str | None = None,
    ) -> Resolution:
        """Walk the ladder for ``model``, then ``fallback`` (request model)."""
        tried: list[str] = []

        for name in (model, fallback):
            if not name:
                continue
            resolution = self._resolve_one(name, provider, tried)
            if resolution.found:
                return resolution

        return Resolution(method=Method.UNPRICED, candidates=tuple(tried))

    def _resolve_one(self, name: str, provider: str | None, tried: list[str]) -> Resolution:
        # Rung 1: exact key.
        tried.append(name)
        if (pricing := self.book.get(name)) is not None:
            return Resolution(Method.EXACT, name, pricing, tuple(tried))

        # Rung 2: provider-qualified, e.g. "gemini" + "gemini-2.5-pro".
        if provider:
            qualified = f"{provider}/{name}"
            tried.append(qualified)
            if (pricing := self.book.get(qualified)) is not None:
                return Resolution(Method.PROVIDER_QUALIFIED, qualified, pricing, tuple(tried))

        # Rung 3: normalized base form, preferring a key from the same provider.
        base = normalize(name)
        candidates = self._normalized_index.get(base, ())
        if candidates:
            chosen = candidates[0]
            if provider:
                for key in candidates:
                    entry = self.book.get(key)
                    if entry is not None and entry.provider == provider:
                        chosen = key
                        break
            tried.append(f"~{base}")
            pricing = self.book.get(chosen)
            if pricing is not None:
                return Resolution(Method.NORMALIZED, chosen, pricing, tuple(tried))

        return Resolution(Method.UNPRICED, candidates=tuple(tried))
