# @tokenomics/sdk

Emit OpenTelemetry GenAI spans with cost attribution to a [Tokenomics](../../README.md) server.

```ts
import OpenAI from "openai";
import { configure, track, wrapOpenAI } from "@tokenomics/sdk";

configure({ endpoint: "http://localhost:8000/v1/traces", project: "checkout", environment: "prod" });
const client = wrapOpenAI(new OpenAI());

await track({ feature: "cart-summarizer", subjectId: "cust_123", promptVersion: "v4" }, async () => {
  await client.chat.completions.create({ model: "gpt-4o", messages: [...] });
});
```

`track()` nests via `AsyncLocalStorage`, so an inner call can add `promptVersion`
without knowing what an outer call set, and concurrent requests never bleed into
each other. Anthropic's exclusive token accounting is converted to the OTel-inclusive
form automatically — see `src/attributes.ts`.

Scope: non-streaming calls only (matching the Python SDK). A streaming response's
usage is only known after the last chunk, which needs a different span lifecycle;
streamed calls fall through to the original method untraced rather than emitting a
wrong span.

Call `flush()` before a short-lived process (a CLI, a Lambda) exits, so buffered
spans are not lost.
