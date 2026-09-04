"""Derived token economics: shares, blended rates, and what the cache is worth."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from tokenomics.finops import tokens as token_service
from tokenomics.finops.tokens import COMPONENTS


def row(**overrides: Any) -> dict[str, Any]:
    """A cache-heavy Anthropic-shaped slice, priced at plausible rates.

    input 3e-6, cache read 3e-7 (0.1x), cache write 3.75e-6 (1.25x), output 1.5e-5.
    """
    base: dict[str, Any] = {
        "requests": 10,
        "cost_usd": Decimal("0.02415"),
        "input_tokens": 10_000,
        "output_tokens": 1_000,
        "cache_read_tokens": 8_000,
        "cache_write_tokens": 1_000,
        "reasoning_tokens": 0,
        "billable_input_tokens": 1_000,
        "billable_output_tokens": 1_000,
        "billable_cache_read_tokens": 8_000,
        "billable_cache_write_tokens": 1_000,
        "billed_reasoning_tokens": 0,
        "input_usd": Decimal("0.003"),
        "output_usd": Decimal("0.015"),
        "cache_read_usd": Decimal("0.0024"),
        "cache_write_usd": Decimal("0.00375"),
        "reasoning_usd": Decimal("0"),
        # 8000 and 1000 cached tokens repriced at the 3e-6 input rate.
        "cache_read_at_input_usd": Decimal("0.024"),
        "cache_write_at_input_usd": Decimal("0.003"),
        "cache_basis_tokens": 9_000,
        "cache_tokens": 9_000,
        "unpriced_events": 0,
    }
    return {**base, **overrides}


def test_total_tokens_is_the_inclusive_total_not_the_sum_of_components():
    """input and output already contain the cache and reasoning counts."""
    derived = token_service.derive(row())
    assert derived["total_tokens"] == 11_000
    # Summing the reported components instead would give 20_000 -- the double count
    # this whole codebase exists to avoid.
    assert derived["total_tokens"] != sum(
        row()[key]
        for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
    )


def test_components_partition_both_the_volume_and_the_bill():
    derived = token_service.derive(row())
    assert sum(derived["token_components"].values()) == derived["total_tokens"]
    assert sum(derived["cost_components"].values()) == derived["cost_usd"]


def test_shares_sum_to_one_on_each_side():
    derived = token_service.derive(row())
    assert sum(derived["token_shares"].values()) == pytest.approx(1.0)
    assert sum(derived["cost_shares"].values()) == pytest.approx(1.0)


def test_the_cheap_bucket_is_most_of_the_volume_and_least_of_the_bill():
    """The headline finding the dashboard exists to show."""
    derived = token_service.derive(row())
    assert derived["token_shares"]["cache_read"] > derived["token_shares"]["output"]
    assert derived["cost_shares"]["output"] > derived["cost_shares"]["cache_read"]


def test_cache_savings_net_off_the_write_premium():
    derived = token_service.derive(row())
    # Reads: would have cost 0.024 at the input rate, actually cost 0.0024.
    assert derived["cache_savings_usd"] == Decimal("0.0216")
    # Writes: cost 0.00375 against a 0.003 counterfactual -- a premium, not a saving.
    assert derived["cache_write_premium_usd"] == Decimal("0.00075")
    assert derived["net_cache_benefit_usd"] == Decimal("0.02085")


def test_a_cache_that_is_rewritten_more_than_it_is_read_is_a_net_loss():
    """Reporting the saving alone would flatter exactly the workload worth catching."""
    derived = token_service.derive(
        row(
            cache_read_tokens=100,
            billable_cache_read_tokens=100,
            cache_write_tokens=9_000,
            billable_cache_write_tokens=9_000,
            cache_read_usd=Decimal("0.00003"),
            cache_write_usd=Decimal("0.03375"),
            cache_read_at_input_usd=Decimal("0.0003"),
            cache_write_at_input_usd=Decimal("0.027"),
        )
    )
    assert derived["net_cache_benefit_usd"] < 0


def test_blended_rate_is_quoted_per_million():
    """A single-component slice makes the unit unambiguous: $15 for 1M output tokens."""
    derived = token_service.derive(
        row(
            cost_usd=Decimal("15"),
            input_tokens=0,
            output_tokens=1_000_000,
            cache_read_tokens=0,
            cache_write_tokens=0,
            billable_input_tokens=0,
            billable_output_tokens=1_000_000,
            billable_cache_read_tokens=0,
            billable_cache_write_tokens=0,
            input_usd=Decimal(0),
            output_usd=Decimal("15"),
            cache_read_usd=Decimal(0),
            cache_write_usd=Decimal(0),
        )
    )
    assert derived["usd_per_1m_tokens"] == Decimal(15)


def test_blended_rate_lies_between_the_cheapest_and_dearest_component():
    derived = token_service.derive(row())
    rates = derived["usd_per_1m_by_component"]
    assert rates["cache_read"] < derived["usd_per_1m_tokens"] < rates["output"]


def test_per_component_rates_recover_the_rates_that_were_billed():
    derived = token_service.derive(row())
    rates = derived["usd_per_1m_by_component"]
    assert rates["input"] == Decimal(3)
    assert rates["output"] == Decimal(15)
    assert rates["cache_read"] == Decimal("0.3")
    assert rates["cache_write"] == Decimal("3.75")


def test_reasoning_priced_apart_gets_its_own_slice():
    derived = token_service.derive(
        row(
            output_tokens=2_000,
            reasoning_tokens=800,
            billable_output_tokens=1_200,
            billed_reasoning_tokens=800,
            output_usd=Decimal("0.018"),
            reasoning_usd=Decimal("0.016"),
            cost_usd=Decimal("0.04315"),
        )
    )
    assert derived["token_components"]["reasoning"] == 800
    assert derived["reasoning_share"] == pytest.approx(0.4)
    assert sum(derived["token_components"].values()) == derived["total_tokens"]


def test_reasoning_bundled_into_output_is_reported_but_not_split_out():
    """Most models have no separate reasoning rate; the tokens are already in output."""
    derived = token_service.derive(
        row(
            output_tokens=2_000,
            reasoning_tokens=800,
            billable_output_tokens=2_000,
            output_usd=Decimal("0.03"),
            cost_usd=Decimal("0.03915"),
        )
    )
    assert derived["token_components"]["reasoning"] == 0
    assert derived["cost_components"]["reasoning"] == Decimal(0)
    # The volume is still visible, just not billed apart.
    assert derived["reasoning_share"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("field", "zeroed"),
    [
        ("cache_hit_rate", {"input_tokens": 0, "cache_read_tokens": 0}),
        ("reasoning_share", {"output_tokens": 0, "reasoning_tokens": 0}),
        ("usd_per_1m_tokens", {"input_tokens": 0, "output_tokens": 0}),
    ],
)
def test_a_ratio_with_no_denominator_is_none_not_zero(field, zeroed):
    """'We have no cached input' and '0% of our input is cached' are different facts."""
    assert token_service.derive(row(**zeroed))[field] is None


def test_partial_rate_coverage_is_reported_rather_than_hidden():
    derived = token_service.derive(row(cache_tokens=10_000, cache_basis_tokens=9_000))
    assert derived["cache_priced_coverage"] == pytest.approx(0.9)


def test_an_empty_row_derives_without_dividing_by_zero():
    derived = token_service.derive({})
    assert derived["total_tokens"] == 0
    assert derived["usd_per_1m_tokens"] is None
    assert derived["net_cache_benefit_usd"] == Decimal(0)
    assert set(derived["token_components"]) == set(COMPONENTS)


def test_summarize_weights_by_volume_not_by_slice():
    """A cheap giant and an expensive speck must not average to the midpoint."""
    giant = row(
        cost_usd=Decimal("1"),
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        billable_input_tokens=1_000_000,
        billable_output_tokens=0,
        billable_cache_read_tokens=0,
        billable_cache_write_tokens=0,
        input_usd=Decimal("1"),
        output_usd=Decimal(0),
        cache_read_usd=Decimal(0),
        cache_write_usd=Decimal(0),
        cache_read_at_input_usd=Decimal(0),
        cache_write_at_input_usd=Decimal(0),
        cache_basis_tokens=0,
        cache_tokens=0,
    )
    speck = {**giant, "cost_usd": Decimal("1"), "input_tokens": 1_000}
    speck = {**speck, "billable_input_tokens": 1_000, "input_usd": Decimal("1")}

    summary = token_service.summarize([giant, speck])
    assert summary["total_tokens"] == 1_001_000
    # Volume-weighted: ~$2 per 1.001M tokens, not the mean of $1/M and $1000/M.
    assert summary["usd_per_1m_tokens"] < Decimal(3)


def test_summarize_drops_subjects_rather_than_double_counting_them():
    """COUNT(DISTINCT) per slice cannot be summed: one customer can be in both."""
    summary = token_service.summarize([row(subjects=10), row(subjects=10)])
    assert "subjects" not in summary


def test_a_slice_with_nothing_priced_reports_an_unknown_rate_not_a_free_one() -> None:
    """`$0.00 per million` would claim the tokens were free, which is the exact
    failure this project exists to prevent -- spend looking better than it is."""
    derived = token_service.derive(
        row(
            requests=10,
            unpriced_events=10,
            cost_usd=Decimal(0),
            input_usd=Decimal(0),
            output_usd=Decimal(0),
            cache_read_usd=Decimal(0),
            cache_write_usd=Decimal(0),
            reasoning_usd=Decimal(0),
        )
    )

    assert derived["total_tokens"] > 0  # the volume is known
    assert derived["usd_per_1m_tokens"] is None  # the rate is not
    assert all(rate is None for rate in derived["usd_per_1m_by_component"].values())
    assert all(share is None for share in derived["cost_shares"].values())


def test_a_partly_unpriced_slice_still_reports_its_rate() -> None:
    """Understated, not unknown -- and `unpriced_events` is what says so."""
    derived = token_service.derive(row(requests=10, unpriced_events=3))

    assert derived["usd_per_1m_tokens"] is not None
    assert derived["unpriced_events"] == 3
