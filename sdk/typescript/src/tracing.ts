/** Tracer setup and span emission. */

import { SpanKind, SpanStatusCode, trace, type Span, type Tracer } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { Resource } from "@opentelemetry/resources";
import { BatchSpanProcessor, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import {
  GEN_AI_OPERATION,
  GEN_AI_PROVIDER,
  GEN_AI_REQUEST_MODEL,
  GEN_AI_RESPONSE_ID,
  GEN_AI_RESPONSE_MODEL,
  usageAttributes,
  type Usage,
} from "./attributes.js";
import { attributionAttributes, current, setBase, attribution, type Attribution } from "./context.js";

export const DEFAULT_ENDPOINT = "http://localhost:8000/v1/traces";
const TRACER_NAME = "tokenomics-sdk";

let provider: NodeTracerProvider | undefined;

export interface ConfigureOptions {
  endpoint?: string;
  project?: string;
  environment?: string;
  serviceName?: string;
  batch?: boolean;
  headers?: Record<string, string>;
}

/** Point the SDK at a Tokenomics endpoint and set default attribution. */
export function configure(options: ConfigureOptions = {}): NodeTracerProvider {
  const endpoint = options.endpoint ?? process.env.TOKENOMICS_ENDPOINT ?? DEFAULT_ENDPOINT;
  setBase(
    attribution({ project: options.project, environment: options.environment }),
  );

  const newProvider = new NodeTracerProvider({
    resource: Resource.default().merge(
      new Resource({ "service.name": options.serviceName ?? "tokenomics-instrumented-app" }),
    ),
  });
  const exporter = new OTLPTraceExporter({ url: endpoint, headers: options.headers });
  const processor =
    options.batch === false ? new SimpleSpanProcessor(exporter) : new BatchSpanProcessor(exporter);
  newProvider.addSpanProcessor(processor);
  newProvider.register();

  provider = newProvider;
  return newProvider;
}

/** Force-export buffered spans. Call before a short-lived process exits. */
export async function flush(): Promise<void> {
  await provider?.forceFlush();
}

export function tracer(): Tracer {
  return trace.getTracer(TRACER_NAME);
}

/** Mutable handle to an in-flight GenAI span. */
export class SpanHandle {
  constructor(private readonly span: Span) {}

  setUsage(value: Usage): void {
    this.span.setAttributes(usageAttributes(value));
  }

  setResponse(options: { model?: string; responseId?: string }): void {
    if (options.model) this.span.setAttribute(GEN_AI_RESPONSE_MODEL, options.model);
    if (options.responseId) this.span.setAttribute(GEN_AI_RESPONSE_ID, options.responseId);
  }

  setAttribute(key: string, value: string | number | boolean): void {
    this.span.setAttribute(key, value);
  }
}

export interface LlmSpanOptions {
  provider: string;
  operation: string;
  requestModel: string;
  extra?: Record<string, string | number | boolean>;
  attribution?: Attribution;
}

/**
 * Wrap an LLM call so the span duration reflects the real call latency. Duration
 * matters downstream -- it feeds cost-per-second and latency-vs-cost views, and a
 * zero-length span would silently break them.
 */
export async function llmSpan<T>(
  options: LlmSpanOptions,
  fn: (span: SpanHandle) => Promise<T>,
): Promise<T> {
  const attributes: Record<string, string | number | boolean> = {
    [GEN_AI_PROVIDER]: options.provider,
    [GEN_AI_OPERATION]: options.operation,
    [GEN_AI_REQUEST_MODEL]: options.requestModel,
    ...attributionAttributes(options.attribution ?? current()),
    ...options.extra,
  };

  return tracer().startActiveSpan(
    `${options.operation} ${options.requestModel}`,
    { kind: SpanKind.CLIENT, attributes },
    async (span) => {
      const handle = new SpanHandle(span);
      try {
        return await fn(handle);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        span.setStatus({ code: SpanStatusCode.ERROR, message });
        span.setAttribute("error.type", error instanceof Error ? error.constructor.name : "Error");
        throw error;
      } finally {
        span.end();
      }
    },
  );
}

export interface RecordCallOptions {
  provider: string;
  operation: string;
  requestModel: string;
  responseModel?: string;
  usage: Usage;
  responseId?: string;
  extra?: Record<string, string | number | boolean>;
}

/** Emit a completed GenAI span in one shot (used by importers and backfills). */
export async function recordCall(options: RecordCallOptions): Promise<void> {
  await llmSpan(
    {
      provider: options.provider,
      operation: options.operation,
      requestModel: options.requestModel,
      extra: options.extra,
    },
    async (span) => {
      span.setResponse({ model: options.responseModel, responseId: options.responseId });
      span.setUsage(options.usage);
    },
  );
}
