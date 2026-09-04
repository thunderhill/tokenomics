# Tokenomics

**FinOps observability and cost governance for multi-LLM applications.**

Tools like Langfuse and LiteLLM will show you *tokens*. Tokenomics turns those tokens
into **money you can govern**: unified spend across every provider and project,
attributed to features and customers, with budgets, forecasts, anomaly detection and
optimization recommendations.

It is deliberately **not** another tracing tool. It consumes the OpenTelemetry GenAI
traces you already emit.

[![GitHub](https://img.shields.io/badge/GitHub-thunderhill%2Ftokenomics-181717?logo=github)](https://github.com/thunderhill/tokenomics)
[![CI](https://github.com/thunderhill/tokenomics/actions/workflows/ci.yml/badge.svg)](https://github.com/thunderhill/tokenomics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

---

## Screenshots

![Tokenomics dashboard](docs/images/dashboard.png)

*Spend, budget burn-down, anomaly detection with probable cause, and cache economics
across a real multi-provider estate.*

Agent harnesses are a first-class source, not just SDK-instrumented apps. `tokenomics
import dsh` reads [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
session logs directly and prices them — correctly landing self-hosted (Ollama) runs as
`UNPRICED` rather than guessing a hosted provider's rate for a same-named model, so a
free local session can still be honestly repriced against a real API via `tokenomics
whatif --provider ollama`.

<p float="left">
  <img src="docs/images/dsh-chat.png" alt="dsh chat" width="49%">
  <img src="docs/images/dsh-trajectory.png" alt="dsh trajectory" width="49%">
</p>

## Why this exists

| Capability | Langfuse | LiteLLM | **Tokenomics** |
|---|:--:|:--:|:--:|
| Token & trace capture | ✅ | ✅ | consumes theirs |
| Per-call cost | ✅ | ✅ | ✅ |
| Cost split by token component | ~ | ✗ | ✅ |
| Value of prompt caching, in dollars | ✗ | ✗ | ✅ |
| Unified spend across providers *and* projects | ~ | ~ | ✅ |
| Attribution to feature / customer / prompt version | ~ | ✗ | ✅ |
| Budgets + threshold alerting | ✗ | ~ | ✅ |
| Month-end forecasting | ✗ | ✗ | ✅ |
| Spend anomaly detection w/ probable cause | ✗ | ✗ | ✅ |
| What-if model-swap simulation | ✗ | ✗ | ✅ |
| Chargeback / showback + unit economics | ✗ | ✗ | ✅ |

## Architecture

```mermaid
flowchart TB
    subgraph apps["Your applications"]
        SDK["tokenomics-sdk (py / ts)<br/>wraps OpenAI + Anthropic"]
        OTEL["any OTel GenAI<br/>instrumentation"]
    end
    subgraph imports["Batch importers"]
        LL["LiteLLM proxy logs"]
        LF["Langfuse export"]
    end

    SDK -->|OTLP/HTTP| ING
    OTEL -->|OTLP/HTTP| ING
    COL["OTel Collector<br/>(gRPC → HTTP)"] -->|OTLP/HTTP| ING
    LL --> IMP["importers"]
    LF --> IMP

    subgraph core["Tokenomics API (FastAPI)"]
        ING["/v1/traces<br/>OTLP ingest"] --> NORM["normalizer<br/>alias table → UsageEvent"]
        IMP --> NORM
        NORM --> PRICE["pricing engine<br/>resolve → partition → cost"]
        PRICE --> STORE[("Postgres 16<br/>partitioned + rollups")]
        STORE --> AGG["attribution / query layer"]
        AGG --> FIN["budgets · forecast<br/>anomalies · what-if"]
        FIN --> REST["REST API + /metrics"]
    end

    PB[("pricebook snapshots<br/>vendored + refreshable")] --> PRICE
    REST --> WEB["React dashboard"]
    FIN -->|HMAC webhooks| ALERT["Slack / PagerDuty / any"]
    REST --> PROM["Prometheus"]
```

## Quick start

```bash
docker compose up -d                      # api + postgres + dashboard
uv run python -m demo.run_demo            # simulate 6 projects, print a cost report
open http://localhost:5173                # dashboard
```

The demo sends 45 days of realistic multi-project traffic as **real OTLP/protobuf**
over HTTP, then prints a chargeback report, the token/cost component split, a month-end
forecast, the anomaly it deliberately injects (with probable cause), budget burn-down
and a what-if model swap.

The six simulated projects are shaped to produce findings, not filler — one whose
reasoning tokens are billed apart at 3.3x its output rate and one whose are bundled
invisibly inside it, one whose prompt cache is a **net loss** because it writes far
more than it reads, and one calling an internal model no price list knows, which lands
as a stated blind spot rather than as free traffic.

## Two correctness properties most cost tools get wrong

Tokenomics is built around two facts that are easy to miss and expensive to get wrong.

### 1. Token counts are inclusive, so cost must *partition*, not sum

The OTel GenAI semantic conventions specify that cache and reasoning counts are
**subsets** of the totals:

```
cache_read + cache_write ⊆ input_tokens
reasoning                ⊆ output_tokens
```

Provider price lists quote **mutually exclusive** per-component rates. Adding the
components together therefore double-counts. Because a cache read is typically
**0.1×** the input rate, a naive implementation overstates a cache-heavy Anthropic
workload by nearly **10×** on its cached portion. Tokenomics partitions instead:

```
billable_input = input − cache_read − cache_write
```

### 2. A model it cannot price is `NULL`, never `$0`

`claude-sonnet-4-5` alone appears under **20 different keys** across Bedrock, Vertex,
Azure, Databricks, Snowflake and regional prefixes, and 2535 of 3111 price-list keys
are namespaced. A naive dictionary lookup silently returns zero — the single worst
failure mode for a cost tool, because your spend looks *better* than it is.

Tokenomics runs an explicit [resolver ladder](src/tokenomics/pricing/resolver.py) and,
when every rung misses, records the event as **unpriced** with `cost = NULL`, increments
`tokenomics_unpriced_events_total`, and surfaces it in the dashboard.

## Token economics

Volume and money are reported together, over the same five disjoint buckets the bill is
actually computed from — because either one alone misleads. A cache read is ~0.1x the
input rate, so the cheapest bucket is usually most of your tokens and almost none of
your spend:

```
component     tokens      of volume   spend     of spend   rate
------------  ----------  ----------  --------  ---------  --------
cache_read    24,264       58.9%      $0.0072     4.6%     $0.30/M
cache_write    3,035        7.4%      $0.0114     7.2%     $3.75/M
input          6,301       15.3%      $0.0180    11.4%     $3.00/M
output         5,810       14.1%      $0.0857    54.1%     $15.00/M
reasoning      1,800        4.4%      $0.0360    22.7%     $20.00/M
```

Optimizing the 59% that is 5% of the bill is the most common wasted quarter in this
field. The same figures are on `/api/spend/tokens`, in `tokenomics tokens`, and on the
dashboard as twin volume/spend bars.

Caching is scored net, not gross:

```
saved by cache reads   $0.0650
premium paid to write  $0.0023      # a cache write is ~1.25x the input rate
net benefit            $0.0628
```

Reporting the saving alone would flatter exactly the workload worth catching — one that
rewrites its cache on every call. Both sides are priced from `rates_applied`, the rates
each event was *actually billed at*, rather than from today's price list: a saving
computed against a rate that has since changed is fiction.

## Attribution

Every event carries tags. Five canonical dimensions are indexed columns; anything else
goes to JSONB.

```python
from openai import OpenAI
from tokenomics_sdk import configure, track, wrap_openai

configure(endpoint="http://localhost:8000/v1/traces", project="checkout", environment="prod")
client = wrap_openai(OpenAI())

with track(feature="cart-summarizer", subject_id="cust_123", prompt_version="v4"):
    client.chat.completions.create(model="gpt-4o", messages=[...])
```

The same surface exists in TypeScript (`sdk/typescript`) — `configure`, `track`,
`wrapOpenAI` / `wrapAnthropic` — using `AsyncLocalStorage` instead of `contextvars` for
the ambient attribution context.

Then query cost by any combination:

```
GET /api/spend/breakdown?group_by=project,feature&since=2026-07-01T00:00:00Z
```

## Pricing data

Pricing comes from LiteLLM's community-maintained model price list, **vendored into
this repo** so the system runs fully offline. Refresh is explicit:

```bash
tokenomics pricing refresh     # the only outbound network call in the product
```

Every snapshot is content-addressed by SHA-256, and each event stores the snapshot id,
resolved model key **and the exact rates applied** — so historical costs never change
when prices do.

## Roadmap

- [x] **v0.1** — OTLP ingest, pricing engine, attribution, budgets, forecasting,
      anomaly detection, what-if, dashboard, Docker Compose
- [x] **v0.1.1** — token/cost component split, reasoning-token accounting, cache
      economics, blended per-1M rates
- [ ] **v0.2** — audio/image token pricing; batch-API discounts; Langfuse + LiteLLM importers
- [ ] **v0.3** — commitment/discount modeling (enterprise agreements, PTUs)
- [ ] **v0.4** — quality-aware what-if (wire real eval scores into model-swap advice)
- [ ] **v0.5** — multi-tenant RBAC, SSO, per-team API keys

## Development

```bash
# API
uv sync
uv run pytest                             # fully offline
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run tokenomics serve                   # http://localhost:8000

# Postgres for local dev (or point TOKENOMICS_DATABASE_URL at your own)
docker run -d -p 55432:5432 -e POSTGRES_USER=tokenomics -e POSTGRES_PASSWORD=tokenomics \
  -e POSTGRES_DB=tokenomics postgres:16-alpine

# Integration tests additionally need a real database
TOKENOMICS_TEST_PG_URL=postgresql://tokenomics:tokenomics@localhost:55432/tokenomics uv run pytest

# Dashboard
cd apps/web && npm install && npm run dev  # http://localhost:5173

# TypeScript SDK
cd sdk/typescript && npm install && npm run build
```

The `tokenomics` CLI covers everything the API does without a server running --
`tokenomics spend`, `tokens`, `forecast`, `anomalies`, `budgets`, `report`, `whatif`,
and `pricing refresh` (the one command that reaches the network).

## License

MIT
