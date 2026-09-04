"""Token economics: what the volume was, and what that volume cost.

Everything here is pure. The SQL sums live in :func:`tokenomics.storage.queries.
token_economics`; this module turns them into the figures people actually reason
about -- component shares, blended rates, and what the prompt cache is worth in
dollars -- which keeps the arithmetic testable without a database, the same split
:mod:`tokenomics.finops.reports` already uses.

Two things to keep straight, both consequences of the inclusive/exclusive mismatch
documented in :mod:`tokenomics.pricing.cost`:

* **The components are disjoint; the totals are not.** ``input_tokens`` already
  contains the cache counts and ``output_tokens`` already contains reasoning, so
  ``total_tokens`` is ``input + output`` -- never the sum of the five components,
  which would double-count. The five components, on the other hand, *do* sum to the
  total, because they are the partition the bill was actually computed from.

* **A ratio with no denominator is ``None``, not zero.** A project with no cached
  input has an unknown cache hit rate, not a 0% one, and the two read very
  differently on a dashboard.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

#: The disjoint billable buckets, in the order they are shown to people: the input
#: side first (cheapest cached reads through to full-rate input), then the output
#: side, which is where the money usually is.
COMPONENTS: tuple[str, ...] = ("cache_read", "cache_write", "input", "output", "reasoning")

#: component -> (token column, cost column). Both sides are the *billed* quantity,
#: so a component's tokens and its dollars always describe the same thing.
_COLUMNS: dict[str, tuple[str, str]] = {
    "cache_read": ("billable_cache_read_tokens", "cache_read_usd"),
    "cache_write": ("billable_cache_write_tokens", "cache_write_usd"),
    "input": ("billable_input_tokens", "input_usd"),
    "output": ("billable_output_tokens", "output_usd"),
    "reasoning": ("billed_reasoning_tokens", "reasoning_usd"),
}

PER_MILLION = Decimal(1_000_000)


def _int(row: dict[str, Any], key: str) -> int:
    return int(row.get(key) or 0)


def _dec(row: dict[str, Any], key: str) -> Decimal:
    value = row.get(key)
    return Decimal(0) if value is None else Decimal(value)


def _ratio(numerator: float | int, denominator: float | int) -> float | None:
    return (numerator / denominator) if denominator else None


def _rate_per_million(cost: Decimal, tokens: int) -> Decimal | None:
    """Effective $/1M tokens -- the unit every provider quotes a price in."""
    return (cost / Decimal(tokens) * PER_MILLION) if tokens else None


def _is_fully_unpriced(row: dict[str, Any]) -> bool:
    """True when nothing in this slice could be priced.

    Cost *sums* coalesce to zero and lean on ``unpriced_events`` to say so, which is
    the established contract. A derived *rate* cannot do the same: "$0.00 per million"
    asserts the tokens were free, which is a stronger and falser claim than "no priced
    spend". So a slice with no priced events at all reports its rates as unknown.
    """
    requests = _int(row, "requests")
    return bool(requests) and _int(row, "unpriced_events") >= requests


def derive(row: dict[str, Any]) -> dict[str, Any]:
    """Add the derived token/cost figures to one aggregated row.

    The input row is not mutated; the returned dict carries the original sums plus
    everything below, so callers can hand it straight to a response model.
    """
    token_components = {name: _int(row, _COLUMNS[name][0]) for name in COMPONENTS}
    cost_components = {name: _dec(row, _COLUMNS[name][1]) for name in COMPONENTS}

    input_tokens = _int(row, "input_tokens")
    output_tokens = _int(row, "output_tokens")
    total_tokens = input_tokens + output_tokens
    cost_usd = _dec(row, "cost_usd")

    billed_tokens = sum(token_components.values())
    billed_cost = sum(cost_components.values(), Decimal(0))

    savings, premium, coverage = _cache_economics(row)
    blind = _is_fully_unpriced(row)

    def rate(cost: Decimal, count: int) -> Decimal | None:
        return None if blind else _rate_per_million(cost, count)

    return {
        **row,
        "total_tokens": total_tokens,
        "token_components": token_components,
        "cost_components": cost_components,
        # Shares are of the billed partition, so each set sums to 1.0 (or is empty).
        "token_shares": {
            name: _ratio(count, billed_tokens) for name, count in token_components.items()
        },
        "cost_shares": {
            name: (float(value / billed_cost) if billed_cost else None)
            for name, value in cost_components.items()
        },
        "usd_per_1m_tokens": rate(cost_usd, total_tokens),
        "usd_per_1m_by_component": {
            name: rate(cost_components[name], token_components[name]) for name in COMPONENTS
        },
        "cache_hit_rate": _ratio(_int(row, "cache_read_tokens"), input_tokens),
        "reasoning_share": _ratio(_int(row, "reasoning_tokens"), output_tokens),
        "cache_savings_usd": savings,
        "cache_write_premium_usd": premium,
        "net_cache_benefit_usd": savings - premium,
        # < 1.0 means some cached traffic could not be priced, so the benefit above is
        # a floor rather than the whole story. None means there was no cached traffic.
        "cache_priced_coverage": coverage,
    }


def _cache_economics(row: dict[str, Any]) -> tuple[Decimal, Decimal, float | None]:
    """What the prompt cache saved, what it cost, and how much of it we could price.

    A cache read is typically 0.1x the input rate and a cache *write* around 1.25x, so
    caching is only a win once reads outnumber writes by enough to clear the premium.
    Reporting the saving alone would flatter a workload that rewrites its cache on
    every call, which is exactly the workload worth catching.
    """
    savings = _dec(row, "cache_read_at_input_usd") - _dec(row, "cache_read_usd")
    premium = _dec(row, "cache_write_usd") - _dec(row, "cache_write_at_input_usd")
    coverage = _ratio(_int(row, "cache_basis_tokens"), _int(row, "cache_tokens"))
    return savings, premium, coverage


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold per-slice rows into one window-wide row, then derive from that.

    Deriving first and averaging afterwards would weight a tiny project the same as a
    large one, so the totals are summed on the raw columns and the ratios computed
    once, at the end.
    """
    totals: dict[str, Any] = {}
    summable = {
        "requests",
        "subjects",
        "unpriced_events",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "billable_input_tokens",
        "billable_output_tokens",
        "billable_cache_read_tokens",
        "billable_cache_write_tokens",
        "billed_reasoning_tokens",
        "cache_basis_tokens",
        "cache_tokens",
        "cost_usd",
        "input_usd",
        "output_usd",
        "cache_read_usd",
        "cache_write_usd",
        "reasoning_usd",
        "cache_read_at_input_usd",
        "cache_write_at_input_usd",
    }
    for row in rows:
        for key in summable:
            value = row.get(key)
            if value is None:
                continue
            current = totals.get(key)
            addend = Decimal(value) if key.endswith("_usd") else int(value)
            totals[key] = addend if current is None else current + addend

    # `subjects` is a COUNT(DISTINCT ...) per slice, so summing it double-counts anyone
    # who appears in two of them. An upper bound is misleading here; drop it.
    totals.pop("subjects", None)
    return derive(totals)
