"""What-if simulation: cache fold-back, per-request tiers, and honest gaps."""

from __future__ import annotations

from decimal import Decimal

from tokenomics.finops import whatif
from tokenomics.finops.whatif import QualityAssumption, Sample, simulate
from tokenomics.models import ServiceTier, TokenVector
from tokenomics.pricing.engine import PricingEngine

#: Priced for input and output but quotes no cache rates at all (verified against the
#: frozen snapshot). Repricing cache-heavy traffic onto it must fold the cache tokens
#: back to the full input rate.
NO_CACHE_MODEL = "mistral/mistral-large-latest"


def cached_sample(requests: int = 100) -> Sample:
    """A cache-heavy call: 10k input of which 9k is a cache read."""
    return Sample(
        tokens=TokenVector(input=10_000, output=500, cache_read=9_000),
        requests=requests,
    )


def test_cache_tokens_fold_back_when_the_target_does_not_price_caching(
    engine: PricingEngine,
) -> None:
    sample = cached_sample()

    kept = simulate(engine, [sample], target_model="claude-sonnet-4-5")
    folded = simulate(engine, [sample], target_model=NO_CACHE_MODEL)

    assert whatif.WARN_CACHE_NOT_PRICED not in kept.warnings
    assert whatif.WARN_CACHE_NOT_PRICED in folded.warnings
    # Reporting the count is the point: it says *why* the target looks expensive.
    assert folded.refolded_cache_tokens == 9_000 * 100
    assert kept.refolded_cache_tokens == 0


def test_the_fold_back_is_priced_at_the_full_input_rate(engine: PricingEngine) -> None:
    # 9000 cached + 1000 uncached input, all billed at Mistral Large's input rate.
    sample = Sample(tokens=TokenVector(input=10_000, output=500, cache_read=9_000))
    uncached = Sample(tokens=TokenVector(input=10_000, output=500))

    folded = simulate(engine, [sample], target_model=NO_CACHE_MODEL)
    plain = simulate(engine, [uncached], target_model=NO_CACHE_MODEL)

    assert folded.projected_usd == plain.projected_usd


def test_a_cheaper_target_shows_a_negative_delta(engine: PricingEngine) -> None:
    baseline_per_call = Decimal("0.01")
    samples = [
        Sample(
            tokens=TokenVector(input=2_000, output=500),
            requests=1_000,
            baseline_usd=baseline_per_call,
        )
    ]

    result = simulate(engine, samples, target_model="gpt-4o-mini")

    assert result.baseline_usd == baseline_per_call * 1_000
    assert result.delta_usd < 0
    assert result.delta_pct is not None
    assert result.delta_pct < 0
    assert result.is_complete


def test_long_context_tiers_are_per_request_not_per_total(engine: PricingEngine) -> None:
    """The reason simulation replays vectors instead of aggregate token totals.

    Threshold-select pricing reprices the *whole request* above 200k input tokens, so
    the same million input tokens costs more as four 250k calls than as ten 100k ones.
    """
    few_big = [Sample(tokens=TokenVector(input=250_000, output=100), requests=4)]
    many_small = [Sample(tokens=TokenVector(input=100_000, output=100), requests=10)]

    assert sum(s.tokens.input * s.requests for s in few_big) == 1_000_000
    assert sum(s.tokens.input * s.requests for s in many_small) == 1_000_000

    big = simulate(engine, few_big, target_model="gemini-2.5-pro")
    small = simulate(engine, many_small, target_model="gemini-2.5-pro")

    assert big.projected_usd > small.projected_usd


def test_requests_beyond_the_target_context_window_are_flagged(engine: PricingEngine) -> None:
    # claude-sonnet-4-5 tops out at 200k input; a 400k prompt would not merely cost
    # more, it would not run. That is a migration blocker, not a rounding note.
    samples = [Sample(tokens=TokenVector(input=400_000, output=500))]
    result = simulate(engine, samples, target_model="claude-sonnet-4-5")
    assert whatif.WARN_CONTEXT_OVERFLOW in result.warnings


def test_an_unresolvable_target_projects_nothing_rather_than_zero_cost(
    engine: PricingEngine,
) -> None:
    samples = [Sample(tokens=TokenVector(input=1_000, output=100), requests=5)]

    result = simulate(engine, samples, target_model="acme/does-not-exist")

    assert result.resolved_model_key is None
    assert result.unpriceable_requests == 5
    assert result.repriced_requests == 0
    # $0 projected with 0 repriced requests reads as "unknown", and is_complete says so.
    assert not result.is_complete


def test_an_unpriced_baseline_is_declared(engine: PricingEngine) -> None:
    samples = [
        Sample(tokens=TokenVector(input=1_000, output=100), requests=3, baseline_usd=None),
        Sample(
            tokens=TokenVector(input=1_000, output=100),
            requests=2,
            baseline_usd=Decimal("0.005"),
        ),
    ]

    result = simulate(engine, samples, target_model="gpt-4o")

    assert result.baseline_unpriced_requests == 3
    assert whatif.WARN_UNPRICED_BASELINE in result.warnings
    assert result.baseline_usd == Decimal("0.010")  # only the priced part
    assert not result.is_complete


def test_sampling_scales_both_sides_and_says_so(engine: PricingEngine) -> None:
    samples = [
        Sample(
            tokens=TokenVector(input=1_000, output=100),
            requests=10,
            baseline_usd=Decimal("0.01"),
        )
    ]

    exact = simulate(engine, samples, target_model="gpt-4o")
    scaled = simulate(engine, samples, target_model="gpt-4o", scale_factor=2.0)

    assert scaled.baseline_usd == exact.baseline_usd * 2
    assert scaled.projected_usd == exact.projected_usd * 2
    assert whatif.WARN_SAMPLED in scaled.warnings
    assert not scaled.is_complete


def test_batch_tier_is_preserved_unless_asked_otherwise(engine: PricingEngine) -> None:
    samples = [
        Sample(
            tokens=TokenVector(input=100_000, output=1_000),
            service_tier=ServiceTier.BATCH,
        )
    ]

    kept = simulate(engine, samples, target_model="gpt-4o")
    standard = simulate(engine, samples, target_model="gpt-4o", keep_service_tier=False)

    assert kept.projected_usd < standard.projected_usd


def test_compare_ranks_by_projection_and_sinks_unpriceable_targets(
    engine: PricingEngine,
) -> None:
    samples = [Sample(tokens=TokenVector(input=10_000, output=1_000), requests=100)]

    results = whatif.compare(
        engine,
        samples,
        targets=["gpt-4o", "gpt-4o-mini", "acme/does-not-exist", "claude-sonnet-4-5"],
    )

    projections = [r.projected_usd for r in results if r.resolved_model_key]
    assert projections == sorted(projections)
    assert results[0].target_model == "gpt-4o-mini"
    assert results[-1].resolved_model_key is None


def test_quality_is_never_inferred(engine: PricingEngine) -> None:
    result = simulate(engine, [cached_sample()], target_model="gpt-4o-mini")
    assert result.quality.status == "not-measured"
    assert result.quality.accepted_quality_delta is None

    annotated = simulate(
        engine,
        [cached_sample()],
        target_model="gpt-4o-mini",
        quality=QualityAssumption(status="measured", evaluation_suite="rag-eval-v3"),
    )
    assert annotated.quality.evaluation_suite == "rag-eval-v3"


def test_the_snapshot_is_recorded_on_every_simulation(
    engine: PricingEngine, frozen_snapshot_sha: str
) -> None:
    # A migration decision must be reproducible against the prices it was made on.
    result = simulate(engine, [cached_sample()], target_model="gpt-4o")
    assert result.snapshot_id == frozen_snapshot_sha


def test_an_empty_body_of_traffic_is_not_a_division_by_zero(engine: PricingEngine) -> None:
    result = simulate(engine, [], target_model="gpt-4o")
    assert result.requests == 0
    assert result.projected_usd == Decimal(0)
    assert result.delta_pct is None
