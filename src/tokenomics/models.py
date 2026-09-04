"""Core domain types for Tokenomics.

The single most important invariant in this package is that OpenTelemetry GenAI
token counts are *inclusive*, not additive:

    cache_read + cache_write <= input          (both are subsets of input)
    reasoning                <= output         (a subset of output)

Provider price lists, by contrast, quote *mutually exclusive* per-component rates.
Costing therefore has to **partition** the totals into disjoint billable buckets
rather than summing the reported components. See ``tokenomics.pricing.cost``.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Component(StrEnum):
    """A disjoint billable token bucket."""

    INPUT = "input"
    OUTPUT = "output"
    CACHE_READ = "cache_read"
    CACHE_WRITE = "cache_write"
    REASONING = "reasoning"


class ServiceTier(StrEnum):
    """Provider service tier, which selects an alternate rate column."""

    STANDARD = "standard"
    PRIORITY = "priority"
    FLEX = "flex"
    BATCH = "batch"


class TokenVector(BaseModel):
    """Reported token counts for a single GenAI call, in semconv (inclusive) form."""

    model_config = ConfigDict(frozen=True)

    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    cache_read: int = Field(default=0, ge=0)
    cache_write: int = Field(default=0, ge=0)
    reasoning: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return self.input + self.output

    def is_consistent(self) -> bool:
        """True when the inclusive-subset invariants hold."""
        return (self.cache_read + self.cache_write) <= self.input and self.reasoning <= self.output


class Attribution(BaseModel):
    """Cost attribution tags.

    The five canonical dimensions are first-class (indexed columns in Postgres);
    anything else lands in ``tags`` (JSONB + GIN index).
    """

    model_config = ConfigDict(frozen=True)

    project: str = "unknown"
    feature: str | None = None
    environment: str | None = None
    subject_id: str | None = None
    prompt_version: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class CostBreakdown(BaseModel):
    """Per-component cost, plus everything needed to audit how it was computed."""

    model_config = ConfigDict(frozen=True)

    total_usd: Decimal
    input_usd: Decimal = Decimal(0)
    output_usd: Decimal = Decimal(0)
    cache_read_usd: Decimal = Decimal(0)
    cache_write_usd: Decimal = Decimal(0)
    reasoning_usd: Decimal = Decimal(0)

    # Audit trail: these travel with the event so historical cost is reproducible
    # even if the pricing snapshot is later deleted.
    model_key: str
    snapshot_id: str
    context_tier: int | None = None
    service_tier: ServiceTier = ServiceTier.STANDARD
    rates_applied: dict[str, Decimal] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def billable_units(self) -> dict[str, Decimal]:
        return {
            "input": self.input_usd,
            "output": self.output_usd,
            "cache_read": self.cache_read_usd,
            "cache_write": self.cache_write_usd,
            "reasoning": self.reasoning_usd,
        }


class UsageEvent(BaseModel):
    """A normalized, provider-agnostic record of one billable GenAI call."""

    model_config = ConfigDict(frozen=True)

    trace_id: str
    span_id: str
    ts: datetime
    duration_ms: float | None = None

    provider: str | None = None
    request_model: str | None = None
    response_model: str | None = None
    operation: str = "chat"
    service_tier: ServiceTier = ServiceTier.STANDARD

    tokens: TokenVector = Field(default_factory=TokenVector)
    attribution: Attribution = Field(default_factory=Attribution)

    cost: CostBreakdown | None = None

    @model_validator(mode="after")
    def _require_a_model(self) -> Self:
        if not (self.request_model or self.response_model):
            msg = "usage event needs request_model or response_model"
            raise ValueError(msg)
        return self

    @property
    def billing_model(self) -> str:
        """The model name to price against: what actually served the request."""
        return self.response_model or self.request_model or ""

    @property
    def is_priced(self) -> bool:
        return self.cost is not None
