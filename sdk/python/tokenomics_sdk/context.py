"""Ambient attribution context.

Attribution is set once at configure time (project, environment) and refined per call
site with :func:`track`, which nests -- so a request handler can set ``feature`` while an
inner helper adds ``prompt_version`` without either knowing about the other.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace

from tokenomics_sdk.attributes import (
    TOKENOMICS_ENVIRONMENT,
    TOKENOMICS_FEATURE,
    TOKENOMICS_PROJECT,
    TOKENOMICS_PROMPT_VERSION,
    TOKENOMICS_SUBJECT,
    TOKENOMICS_TAG_PREFIX,
)


@dataclass(frozen=True, slots=True)
class Attribution:
    project: str | None = None
    feature: str | None = None
    environment: str | None = None
    subject_id: str | None = None
    prompt_version: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def merge(self, other: Attribution) -> Attribution:
        """Overlay non-``None`` fields of ``other``; tags union with ``other`` winning."""
        return Attribution(
            project=other.project or self.project,
            feature=other.feature or self.feature,
            environment=other.environment or self.environment,
            subject_id=other.subject_id or self.subject_id,
            prompt_version=other.prompt_version or self.prompt_version,
            tags={**self.tags, **other.tags},
        )

    def as_attributes(self) -> dict[str, str]:
        mapping = {
            TOKENOMICS_PROJECT: self.project,
            TOKENOMICS_FEATURE: self.feature,
            TOKENOMICS_ENVIRONMENT: self.environment,
            TOKENOMICS_SUBJECT: self.subject_id,
            TOKENOMICS_PROMPT_VERSION: self.prompt_version,
        }
        attributes = {key: value for key, value in mapping.items() if value is not None}
        attributes.update(
            {f"{TOKENOMICS_TAG_PREFIX}{key}": value for key, value in self.tags.items()}
        )
        return attributes


_BASE = Attribution()
# Default is None rather than a shared Attribution instance: the dataclass carries a
# mutable ``tags`` dict, and one shared default could leak tags between contexts.
_CURRENT: ContextVar[Attribution | None] = ContextVar("tokenomics_attribution", default=None)


def set_base(attribution: Attribution) -> None:
    """Set process-wide defaults (called by :func:`tokenomics_sdk.configure`)."""
    global _BASE
    _BASE = attribution


def current() -> Attribution:
    return _BASE.merge(_CURRENT.get() or Attribution())


@contextmanager
def track(
    *,
    project: str | None = None,
    feature: str | None = None,
    environment: str | None = None,
    subject_id: str | None = None,
    prompt_version: str | None = None,
    **tags: str,
) -> Iterator[Attribution]:
    """Attribute every LLM call made inside this block.

    Nests: inner values override outer ones, everything else is inherited.
    """
    overlay = Attribution(
        project=project,
        feature=feature,
        environment=environment,
        subject_id=subject_id,
        prompt_version=prompt_version,
        tags={key: str(value) for key, value in tags.items()},
    )
    merged = (_CURRENT.get() or Attribution()).merge(overlay)
    token = _CURRENT.set(merged)
    try:
        yield merged
    finally:
        _CURRENT.reset(token)


def with_tags(**tags: str) -> Attribution:
    return replace(current(), tags={**current().tags, **tags})
