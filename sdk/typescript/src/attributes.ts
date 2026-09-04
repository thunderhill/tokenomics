/**
 * Attribute names and provider usage-shape adapters.
 *
 * The provider asymmetry that matters: OpenTelemetry GenAI defines
 * `gen_ai.usage.input_tokens` as *inclusive* of cached tokens. Providers disagree
 * about this at the API level.
 *
 * - **OpenAI** already reports inclusively -- `prompt_tokens` contains
 *   `prompt_tokens_details.cached_tokens`, and `completion_tokens` contains
 *   `completion_tokens_details.reasoning_tokens`. Pass them through unchanged.
 * - **Anthropic** reports *exclusively* -- `input_tokens` counts only uncached
 *   input, with `cache_read_input_tokens` and `cache_creation_input_tokens`
 *   reported *alongside* it, not inside it.
 *
 * So Anthropic usage is converted to the inclusive form on the way out:
 * `input = input_tokens + cache_read + cache_creation`. Getting this backwards
 * understates a cache-heavy Anthropic workload's input by the entire cached
 * portion -- which for a well-cached agent is most of the prompt.
 */

export const GEN_AI_PROVIDER = "gen_ai.provider.name";
export const GEN_AI_OPERATION = "gen_ai.operation.name";
export const GEN_AI_REQUEST_MODEL = "gen_ai.request.model";
export const GEN_AI_RESPONSE_MODEL = "gen_ai.response.model";
export const GEN_AI_RESPONSE_ID = "gen_ai.response.id";

export const USAGE_INPUT = "gen_ai.usage.input_tokens";
export const USAGE_OUTPUT = "gen_ai.usage.output_tokens";
export const USAGE_CACHE_READ = "gen_ai.usage.cache_read.input_tokens";
export const USAGE_CACHE_WRITE = "gen_ai.usage.cache_write.input_tokens";
export const USAGE_REASONING = "gen_ai.usage.reasoning.output_tokens";

export const TOKENOMICS_PROJECT = "tokenomics.project";
export const TOKENOMICS_FEATURE = "tokenomics.feature";
export const TOKENOMICS_ENVIRONMENT = "tokenomics.environment";
export const TOKENOMICS_SUBJECT = "tokenomics.subject_id";
export const TOKENOMICS_PROMPT_VERSION = "tokenomics.prompt_version";
export const TOKENOMICS_TAG_PREFIX = "tokenomics.tag.";

export interface Usage {
  readonly input: number;
  readonly output: number;
  readonly cacheRead: number;
  readonly cacheWrite: number;
  readonly reasoning: number;
}

export function usage(partial: Partial<Usage> = {}): Usage {
  return {
    input: partial.input ?? 0,
    output: partial.output ?? 0,
    cacheRead: partial.cacheRead ?? 0,
    cacheWrite: partial.cacheWrite ?? 0,
    reasoning: partial.reasoning ?? 0,
  };
}

export function usageAttributes(value: Usage): Record<string, number> {
  const attributes: Record<string, number> = {
    [USAGE_INPUT]: value.input,
    [USAGE_OUTPUT]: value.output,
  };
  if (value.cacheRead) attributes[USAGE_CACHE_READ] = value.cacheRead;
  if (value.cacheWrite) attributes[USAGE_CACHE_WRITE] = value.cacheWrite;
  if (value.reasoning) attributes[USAGE_REASONING] = value.reasoning;
  return attributes;
}

// Provider SDK response shapes are read structurally so this package never depends
// on `openai` or `@anthropic-ai/sdk` -- the same reason the Python wrappers patch
// bound methods instead of subclassing.
interface OpenAiUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
  completion_tokens_details?: { reasoning_tokens?: number };
}

interface AnthropicUsage {
  input_tokens?: number;
  output_tokens?: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}

function num(value: number | undefined): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/** OpenAI already reports inclusively; pass through. */
export function usageFromOpenAI(raw: OpenAiUsage | null | undefined): Usage {
  if (!raw) return usage();
  return usage({
    input: num(raw.prompt_tokens),
    output: num(raw.completion_tokens),
    cacheRead: num(raw.prompt_tokens_details?.cached_tokens),
    reasoning: num(raw.completion_tokens_details?.reasoning_tokens),
  });
}

/** Anthropic reports exclusively; fold cache counts back into the input total. */
export function usageFromAnthropic(raw: AnthropicUsage | null | undefined): Usage {
  if (!raw) return usage();
  const uncachedInput = num(raw.input_tokens);
  const cacheRead = num(raw.cache_read_input_tokens);
  const cacheWrite = num(raw.cache_creation_input_tokens);
  return usage({
    input: uncachedInput + cacheRead + cacheWrite,
    output: num(raw.output_tokens),
    cacheRead,
    cacheWrite,
  });
}
