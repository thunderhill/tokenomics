"""Tokenomics SDK -- emit cost-attributed OpenTelemetry GenAI spans.

from openai import OpenAI
from tokenomics_sdk import configure, track, wrap_openai

configure(endpoint="http://localhost:8000/v1/traces", project="checkout")
client = wrap_openai(OpenAI())

with track(feature="cart-summarizer", subject_id="cust_123", prompt_version="v4"):
    client.chat.completions.create(model="gpt-4o", messages=[...])
"""

from tokenomics_sdk.attributes import Usage, usage_from_anthropic, usage_from_openai
from tokenomics_sdk.context import Attribution, track
from tokenomics_sdk.tracing import configure, flush, llm_span, record_call
from tokenomics_sdk.wrappers import wrap_anthropic, wrap_openai

__all__ = [
    "Attribution",
    "Usage",
    "configure",
    "flush",
    "llm_span",
    "record_call",
    "track",
    "usage_from_anthropic",
    "usage_from_openai",
    "wrap_anthropic",
    "wrap_openai",
]
