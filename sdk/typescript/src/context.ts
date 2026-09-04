/**
 * Ambient attribution context.
 *
 * Attribution is set once at configure time (project, environment) and refined per
 * call site with `track`, which nests via `AsyncLocalStorage` -- so a request handler
 * can set `feature` while an inner `await`-ed helper adds `promptVersion` without
 * either knowing about the other, and concurrent requests never bleed into each other.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import {
  TOKENOMICS_ENVIRONMENT,
  TOKENOMICS_FEATURE,
  TOKENOMICS_PROJECT,
  TOKENOMICS_PROMPT_VERSION,
  TOKENOMICS_SUBJECT,
  TOKENOMICS_TAG_PREFIX,
} from "./attributes.js";

export interface Attribution {
  readonly project?: string;
  readonly feature?: string;
  readonly environment?: string;
  readonly subjectId?: string;
  readonly promptVersion?: string;
  readonly tags: Readonly<Record<string, string>>;
}

export function attribution(partial: Partial<Attribution> = {}): Attribution {
  return {
    project: partial.project,
    feature: partial.feature,
    environment: partial.environment,
    subjectId: partial.subjectId,
    promptVersion: partial.promptVersion,
    tags: partial.tags ?? {},
  };
}

/** Overlay the non-undefined fields of `overlay`; tags union with `overlay` winning. */
export function mergeAttribution(base: Attribution, overlay: Attribution): Attribution {
  return {
    project: overlay.project ?? base.project,
    feature: overlay.feature ?? base.feature,
    environment: overlay.environment ?? base.environment,
    subjectId: overlay.subjectId ?? base.subjectId,
    promptVersion: overlay.promptVersion ?? base.promptVersion,
    tags: { ...base.tags, ...overlay.tags },
  };
}

export function attributionAttributes(value: Attribution): Record<string, string> {
  const mapping: Record<string, string | undefined> = {
    [TOKENOMICS_PROJECT]: value.project,
    [TOKENOMICS_FEATURE]: value.feature,
    [TOKENOMICS_ENVIRONMENT]: value.environment,
    [TOKENOMICS_SUBJECT]: value.subjectId,
    [TOKENOMICS_PROMPT_VERSION]: value.promptVersion,
  };
  const attributes: Record<string, string> = {};
  for (const [key, val] of Object.entries(mapping)) {
    if (val !== undefined) attributes[key] = val;
  }
  for (const [key, val] of Object.entries(value.tags)) {
    attributes[`${TOKENOMICS_TAG_PREFIX}${key}`] = val;
  }
  return attributes;
}

let base: Attribution = attribution();
const storage = new AsyncLocalStorage<Attribution>();

/** Set process-wide defaults (called by {@link configure}). */
export function setBase(value: Attribution): void {
  base = value;
}

export function current(): Attribution {
  return mergeAttribution(base, storage.getStore() ?? attribution());
}

export interface TrackOptions {
  project?: string;
  feature?: string;
  environment?: string;
  subjectId?: string;
  promptVersion?: string;
  tags?: Record<string, string>;
}

/**
 * Attribute every LLM call made inside `fn`. Nests: inner values override outer
 * ones, everything else is inherited. Works across `await` because it rides
 * `AsyncLocalStorage`, not a plain variable.
 */
export function track<T>(options: TrackOptions, fn: () => T): T {
  const overlay = attribution({
    project: options.project,
    feature: options.feature,
    environment: options.environment,
    subjectId: options.subjectId,
    promptVersion: options.promptVersion,
    tags: options.tags,
  });
  const merged = mergeAttribution(storage.getStore() ?? attribution(), overlay);
  return storage.run(merged, fn);
}
