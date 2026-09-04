/**
 * Tokenomics SDK -- emit cost-attributed OpenTelemetry GenAI spans.
 *
 * ```ts
 * import OpenAI from "openai";
 * import { configure, track, wrapOpenAI } from "@tokenomics/sdk";
 *
 * configure({ endpoint: "http://localhost:8000/v1/traces", project: "checkout" });
 * const client = wrapOpenAI(new OpenAI());
 *
 * await track({ feature: "cart-summarizer", subjectId: "cust_123", promptVersion: "v4" }, async () => {
 *   await client.chat.completions.create({ model: "gpt-4o", messages: [...] });
 * });
 * ```
 */

export {
  usage,
  usageAttributes,
  usageFromAnthropic,
  usageFromOpenAI,
  type Usage,
} from "./attributes.js";
export {
  attribution,
  current,
  track,
  type Attribution,
  type TrackOptions,
} from "./context.js";
export {
  configure,
  flush,
  llmSpan,
  recordCall,
  tracer,
  SpanHandle,
  type ConfigureOptions,
  type LlmSpanOptions,
  type RecordCallOptions,
} from "./tracing.js";
export { wrapAnthropic, wrapOpenAI } from "./wrappers.js";
