/**
 * Drop-in wrappers for the OpenAI and Anthropic SDK clients.
 *
 * The wrappers are structural, not subclass-based: they patch the bound `create`
 * method on the client instance. That keeps the original client object -- with all
 * of its typing, helpers and configuration -- fully intact, and means this package
 * never needs `openai` or `@anthropic-ai/sdk` as a dependency.
 *
 * Scope: non-streaming calls only, matching the Python SDK. A streamed response's
 * usage is only known after the last chunk, which needs a different span lifecycle;
 * v0.2 territory, not silently mishandled here (streaming calls fall through to the
 * original method, untraced, rather than emitting a wrong span).
 */

import { usageFromAnthropic, usageFromOpenAI, type Usage } from "./attributes.js";
import { llmSpan } from "./tracing.js";

const WRAPPED = Symbol.for("tokenomics.wrapped");

interface Wrappable {
  (...args: unknown[]): unknown;
  [WRAPPED]?: boolean;
}

function isStreamingRequest(args: unknown[]): boolean {
  const body = args[0];
  return typeof body === "object" && body !== null && (body as { stream?: unknown }).stream === true;
}

function field(obj: unknown, name: string): unknown {
  return obj && typeof obj === "object" ? (obj as Record<string, unknown>)[name] : undefined;
}

function instrument<U>(
  owner: Record<string, unknown>,
  methodName: string,
  options: {
    provider: string;
    operation: string;
    extractUsage: (raw: U) => Usage;
  },
): void {
  const original = owner[methodName] as Wrappable | undefined;
  if (typeof original !== "function" || original[WRAPPED]) return;

  const wrapper: Wrappable = async function (this: unknown, ...args: unknown[]) {
    if (isStreamingRequest(args)) {
      return original.apply(this, args);
    }
    const body = args[0] as Record<string, unknown> | undefined;
    const requestModel = typeof body?.model === "string" ? body.model : "unknown";

    return llmSpan({ provider: options.provider, operation: options.operation, requestModel }, async (span) => {
      const response = await original.apply(this, args);
      const usage = field(response, "usage");
      if (usage !== undefined) {
        span.setUsage(options.extractUsage(usage as U));
      }
      const model = field(response, "model");
      const id = field(response, "id");
      span.setResponse({
        model: typeof model === "string" ? model : undefined,
        responseId: typeof id === "string" ? id : undefined,
      });
      return response;
    });
  };
  wrapper[WRAPPED] = true;
  owner[methodName] = wrapper;
}

// Minimal structural shape of the parts of the OpenAI client this SDK touches.
interface OpenAiLike {
  chat: { completions: Record<string, unknown> };
  responses?: Record<string, unknown>;
}

/** Instrument an OpenAI client in place and return it. */
export function wrapOpenAI<T extends OpenAiLike>(client: T): T {
  instrument(client.chat.completions, "create", {
    provider: "openai",
    operation: "chat",
    extractUsage: usageFromOpenAI,
  });
  if (client.responses) {
    instrument(client.responses, "create", {
      provider: "openai",
      operation: "chat",
      extractUsage: usageFromOpenAI,
    });
  }
  return client;
}

// Minimal structural shape of the parts of the Anthropic client this SDK touches.
interface AnthropicLike {
  messages: Record<string, unknown>;
}

/** Instrument an Anthropic client in place and return it. */
export function wrapAnthropic<T extends AnthropicLike>(client: T): T {
  instrument(client.messages, "create", {
    provider: "anthropic",
    operation: "chat",
    extractUsage: usageFromAnthropic,
  });
  return client;
}
