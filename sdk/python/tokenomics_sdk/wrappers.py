"""Drop-in wrappers for the OpenAI and Anthropic clients.

The wrappers are structural, not subclass-based: they patch the bound ``create`` method
on the client instance. That keeps the original client object -- with all of its typing,
helpers and configuration -- fully intact, and means the SDK does not need the provider
packages installed to import.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from tokenomics_sdk.attributes import Usage, usage_from_anthropic, usage_from_openai
from tokenomics_sdk.tracing import llm_span


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _instrument(
    owner: Any,
    method_name: str,
    *,
    provider: str,
    operation: str,
    extract_usage: Callable[[Any], Usage],
) -> None:
    """Replace ``owner.method_name`` with a span-emitting wrapper (idempotent)."""
    original = getattr(owner, method_name)
    if getattr(original, "__tokenomics_wrapped__", False):
        return

    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        request_model = str(kwargs.get("model", "unknown"))
        with llm_span(provider=provider, operation=operation, request_model=request_model) as span:
            response = original(*args, **kwargs)
            usage = _field(response, "usage")
            if usage is not None:
                span.set_usage(extract_usage(usage))
            span.set_response(
                model=_field(response, "model"),
                response_id=_field(response, "id"),
            )
            return response

    wrapper.__tokenomics_wrapped__ = True  # type: ignore[attr-defined]
    setattr(owner, method_name, wrapper)


def wrap_openai[ClientT](client: ClientT) -> ClientT:
    """Instrument an OpenAI client in place and return it."""
    completions = client.chat.completions  # type: ignore[attr-defined]
    _instrument(
        completions,
        "create",
        provider="openai",
        operation="chat",
        extract_usage=usage_from_openai,
    )

    responses = getattr(client, "responses", None)
    if responses is not None:
        _instrument(
            responses,
            "create",
            provider="openai",
            operation="chat",
            extract_usage=usage_from_openai,
        )
    return client


def wrap_anthropic[ClientT](client: ClientT) -> ClientT:
    """Instrument an Anthropic client in place and return it."""
    messages = client.messages  # type: ignore[attr-defined]
    _instrument(
        messages,
        "create",
        provider="anthropic",
        operation="chat",
        extract_usage=usage_from_anthropic,
    )
    return client
