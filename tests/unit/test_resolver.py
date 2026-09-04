from __future__ import annotations

import pytest

from tokenomics.models import TokenVector
from tokenomics.pricing.engine import PricingEngine
from tokenomics.pricing.resolver import Method, ModelResolver, normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"),
        ("eu.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"),
        ("bedrock/us-gov-east-1/anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"),
        ("vertex_ai/claude-sonnet-4-5@20250929", "claude-sonnet-4-5"),
        ("databricks/databricks-claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("perplexity/anthropic/claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("gpt-4o-2024-08-06", "gpt-4o"),
    ],
)
def test_normalize_collapses_platform_variants(raw, expected):
    assert normalize(raw) == expected


def test_every_sonnet_45_alias_collapses_to_one_base(book):
    aliases = [k for k in book.models if "sonnet-4-5" in k and "sonnet-4-5-v" not in k]
    assert len(aliases) >= 20
    assert {normalize(k) for k in aliases} == {"claude-sonnet-4-5"}


def test_every_alias_resolves_to_something_priceable(book):
    """No alias may fall through to unpriced -- that is how spend silently vanishes."""
    resolver = ModelResolver(book)
    aliases = [k for k in book.models if "sonnet-4-5" in k]
    for alias in aliases:
        resolution = resolver.resolve(alias, provider="anthropic")
        assert resolution.found, alias
        assert resolution.pricing is not None


def test_exact_match_wins_over_normalization(book):
    """Bedrock and first-party prices genuinely differ, so exact keys must not collapse."""
    resolver = ModelResolver(book)
    key = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    resolution = resolver.resolve(key, provider="bedrock")
    assert resolution.method is Method.EXACT
    assert resolution.model_key == key


def test_unknown_model_is_unpriced_not_zero(engine: PricingEngine):
    cost, resolution = engine.price_tokens(
        TokenVector(input=1000, output=1000), model="definitely-not-a-real-model-9000"
    )
    assert cost is None
    assert resolution.method is Method.UNPRICED
    assert resolution.model_key is None


def test_unpriced_records_what_it_tried(engine: PricingEngine):
    _, resolution = engine.price_tokens(
        TokenVector(input=1), model="mystery-model", provider="acme"
    )
    assert "mystery-model" in resolution.candidates
    assert "acme/mystery-model" in resolution.candidates


def test_provider_qualified_rung(book):
    """A bare name that only exists namespaced must resolve via the provider hint."""
    resolver = ModelResolver(book)
    namespaced = next(
        k for k in book.models if k.startswith("mistral/") and book.models[k].is_priceable
    )
    bare = namespaced.split("/", 1)[1]
    resolution = resolver.resolve(bare, provider="mistral")
    assert resolution.found
    assert resolution.method in {Method.EXACT, Method.PROVIDER_QUALIFIED, Method.NORMALIZED}


def test_falls_back_from_response_model_to_request_model(engine: PricingEngine):
    cost, resolution = engine.price_tokens(
        TokenVector(input=1000, output=100),
        model="hallucinated-response-model",
        fallback_model="gpt-4o",
    )
    assert cost is not None
    assert resolution.model_key == "gpt-4o"


def test_self_hosted_provider_is_unpriced_not_a_cross_provider_guess(book):
    """Regression guard for the dsh/Ollama integration's whole premise.

    Before this fix, `resolve("deepseek-r1", provider="ollama")` picked whatever key the
    normalized index happened to sort first among same-named candidates -- in the real
    vendored book, `azure_ai/deepseek-r1` at $1.35/M input. A free local run would have been
    silently billed at Azure's rate. There is no `ollama/deepseek-r1` key (Ollama's own book
    entries are separate, unrelated models, all priced $0), so this must resolve UNPRICED.
    """
    resolver = ModelResolver(book)

    resolution = resolver.resolve("deepseek-r1", provider="ollama")
    assert resolution.method is Method.UNPRICED
    assert resolution.model_key is None
    assert resolution.model_key != "azure_ai/deepseek-r1"


def test_self_hosted_provider_still_resolves_its_own_book_entry(book):
    """The guard only refuses a *cross*-provider guess -- a real `ollama/*` key still wins."""
    resolver = ModelResolver(book)
    ollama_keys = [k for k in book.models if k.startswith("ollama/")]
    assert ollama_keys, "fixture assumption: the vendored book carries ollama/* entries"
    bare = ollama_keys[0].removeprefix("ollama/")

    resolution = resolver.resolve(bare, provider="ollama")
    assert resolution.found
    assert resolution.pricing is not None
    assert resolution.pricing.provider == "ollama"


def test_non_self_hosted_provider_keeps_the_cross_provider_fallback(book):
    """The guard is scoped to self-hosted providers; every other provider is unchanged."""
    resolver = ModelResolver(book)
    # A provider that is not self-hosted and does not itself carry this model should still
    # fall through to the normalized "plainest key" rung, exactly as before this change.
    resolution = resolver.resolve("deepseek-r1", provider="some-unlisted-gateway")
    assert resolution.method is Method.NORMALIZED
    assert resolution.found
