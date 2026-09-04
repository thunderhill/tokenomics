"""Attribute names and provider usage-shape adapters.

**The provider asymmetry that matters.** OpenTelemetry GenAI defines
``gen_ai.usage.input_tokens`` as *inclusive* of cached tokens. Providers disagree about
this at the API level:

* **OpenAI** already reports inclusively -- ``prompt_tokens`` contains
  ``prompt_tokens_details.cached_tokens``, and ``completion_tokens`` contains
  ``completion_tokens_details.reasoning_tokens``. Pass them through unchanged.

* **Anthropic** reports *exclusively* -- ``input_tokens`` counts only uncached input,
  with ``cache_read_input_tokens`` and ``cache_creation_input_tokens`` reported
  **alongside** it, not inside it.

So Anthropic usage must be converted to the inclusive form on the way out::

    gen_ai.usage.input_tokens = input_tokens + cache_read + cache_creation

Getting this backwards understates a cache-heavy Anthropic workload's input by the
entire cached portion -- which for a well-cached agent is most of the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GEN_AI_PROVIDER = "gen_ai.provider.name"
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"

USAGE_INPUT = "gen_ai.usage.input_tokens"
USAGE_OUTPUT = "gen_ai.usage.output_tokens"
USAGE_CACHE_READ = "gen_ai.usage.cache_read.input_tokens"
USAGE_CACHE_WRITE = "gen_ai.usage.cache_write.input_tokens"
USAGE_REASONING = "gen_ai.usage.reasoning.output_tokens"

TOKENOMICS_PROJECT = "tokenomics.project"
TOKENOMICS_FEATURE = "tokenomics.feature"
TOKENOMICS_ENVIRONMENT = "tokenomics.environment"
TOKENOMICS_SUBJECT = "tokenomics.subject_id"
TOKENOMICS_PROMPT_VERSION = "tokenomics.prompt_version"
TOKENOMICS_TAG_PREFIX = "tokenomics.tag."


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage in OTel GenAI (inclusive) form."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0

    def as_attributes(self) -> dict[str, int]:
        attributes = {USAGE_INPUT: self.input, USAGE_OUTPUT: self.output}
        if self.cache_read:
            attributes[USAGE_CACHE_READ] = self.cache_read
        if self.cache_write:
            attributes[USAGE_CACHE_WRITE] = self.cache_write
        if self.reasoning:
            attributes[USAGE_REASONING] = self.reasoning
        return attributes


def _get(obj: Any, name: str, default: int = 0) -> int:
    """Read a field from a pydantic model, dataclass or plain dict."""
    if obj is None:
        return default
    value = obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    return int(value) if isinstance(value, int | float) else default


def _sub(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def usage_from_openai(usage: Any) -> Usage:
    """OpenAI already reports inclusively; pass through."""
    return Usage(
        input=_get(usage, "prompt_tokens"),
        output=_get(usage, "completion_tokens"),
        cache_read=_get(_sub(usage, "prompt_tokens_details"), "cached_tokens"),
        reasoning=_get(_sub(usage, "completion_tokens_details"), "reasoning_tokens"),
    )


def usage_from_anthropic(usage: Any) -> Usage:
    """Anthropic reports exclusively; fold cache counts back into the input total."""
    uncached_input = _get(usage, "input_tokens")
    cache_read = _get(usage, "cache_read_input_tokens")
    cache_write = _get(usage, "cache_creation_input_tokens")
    return Usage(
        input=uncached_input + cache_read + cache_write,
        output=_get(usage, "output_tokens"),
        cache_read=cache_read,
        cache_write=cache_write,
    )
