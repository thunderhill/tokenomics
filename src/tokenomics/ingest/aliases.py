"""Versioned attribute alias table for OpenTelemetry GenAI spans.

The GenAI semantic conventions are actively churning. During development of this project
the entire convention **moved repositories** -- from ``open-telemetry/semantic-conventions``
(where ``model/gen-ai`` is now marked deprecated) to
``open-telemetry/semantic-conventions-genai`` -- and attributes were renamed along the way::

    gen_ai.usage.cache_creation.input_tokens  ->  gen_ai.usage.cache_write.input_tokens

Real fleets run a mix of instrumentation versions simultaneously, so hardcoding one
generation's names would silently drop usage from the others -- and dropped usage means
understated spend. Every attribute is therefore looked up through an ordered alias list.

Order matters: the first name present wins, so canonical names precede deprecated ones.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Arbitrary user tags ride on this prefix and land in the JSONB ``tags`` column.
TAG_PREFIX = "tokenomics.tag."

USAGE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "input": (
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.prompt_tokens",  # deprecated (pre-1.27)
        "llm.usage.prompt_tokens",  # OpenLLMetry
        "llm.token_count.prompt",  # OpenInference
    ),
    "output": (
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.completion_tokens",  # deprecated (pre-1.27)
        "llm.usage.completion_tokens",
        "llm.token_count.completion",
    ),
    "cache_read": (
        "gen_ai.usage.cache_read.input_tokens",
        "gen_ai.usage.text.cache_read.input_tokens",
        "llm.token_count.prompt_details.cache_read",
    ),
    "cache_write": (
        "gen_ai.usage.cache_write.input_tokens",  # semantic-conventions-genai
        "gen_ai.usage.cache_creation.input_tokens",  # renamed; still widely emitted
        "llm.token_count.prompt_details.cache_write",
    ),
    "reasoning": (
        "gen_ai.usage.reasoning.output_tokens",
        "llm.token_count.completion_details.reasoning",
    ),
}

IDENTITY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "provider": (
        "gen_ai.provider.name",
        "gen_ai.system",  # deprecated
        "llm.system",
    ),
    "request_model": ("gen_ai.request.model", "llm.model_name", "llm.request.model"),
    "response_model": ("gen_ai.response.model", "llm.response.model"),
    "operation": ("gen_ai.operation.name",),
    "service_tier": (
        "gen_ai.openai.response.service_tier",
        "gen_ai.openai.request.service_tier",
    ),
}

ATTRIBUTION_ALIASES: Mapping[str, tuple[str, ...]] = {
    "project": ("tokenomics.project", "project", "service.name"),
    "feature": ("tokenomics.feature", "feature", "gen_ai.workflow.name"),
    "environment": (
        "tokenomics.environment",
        "environment",
        "deployment.environment.name",
        "deployment.environment",
    ),
    "subject_id": ("tokenomics.subject_id", "user.id", "enduser.id", "gen_ai.conversation.id"),
    "prompt_version": ("tokenomics.prompt_version", "gen_ai.prompt.version", "gen_ai.prompt.name"),
}


def first_present(attributes: Mapping[str, Any], names: tuple[str, ...]) -> Any | None:
    """Return the value of the first alias present, preserving alias precedence."""
    for name in names:
        if (value := attributes.get(name)) is not None:
            return value
    return None


def coerce_int(value: Any) -> int:
    """Token counts arrive as ints, floats or numeric strings depending on the exporter."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(float(value)), 0)
        except ValueError:
            return 0
    return 0
