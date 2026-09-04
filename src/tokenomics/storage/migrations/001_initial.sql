-- Tokenomics initial schema.
--
-- Design notes:
--   * usage_event is RANGE-partitioned monthly on ts. Postgres requires the partition
--     key in the primary key, which is fine here: (ts, trace_id, span_id) is also
--     exactly the idempotency key, so collector retries are free of double-billing.
--   * cost_usd is NULLABLE ON PURPOSE. NULL means "we could not price this", which is
--     a very different fact from "this was free". Never write 0 for an unpriced event.
--   * Attribution is hybrid: five canonical dimensions are indexed columns for fast
--     GROUP BY; arbitrary extras live in tags JSONB with a GIN index.

CREATE TABLE IF NOT EXISTS pricing_snapshot (
    snapshot_id  text PRIMARY KEY,
    source       text        NOT NULL,
    fetched_at   timestamptz NOT NULL,
    model_count  integer     NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS usage_event (
    ts                  timestamptz NOT NULL,
    trace_id            text        NOT NULL,
    span_id             text        NOT NULL,
    duration_ms         double precision,

    provider            text,
    request_model       text,
    response_model      text,
    model_key           text,
    operation           text        NOT NULL DEFAULT 'chat',
    service_tier        text        NOT NULL DEFAULT 'standard',

    input_tokens        bigint      NOT NULL DEFAULT 0,
    output_tokens       bigint      NOT NULL DEFAULT 0,
    cache_read_tokens   bigint      NOT NULL DEFAULT 0,
    cache_write_tokens  bigint      NOT NULL DEFAULT 0,
    reasoning_tokens    bigint      NOT NULL DEFAULT 0,

    -- NULL = unpriced. See note above.
    cost_usd            numeric(24, 12),
    input_usd           numeric(24, 12),
    output_usd          numeric(24, 12),
    cache_read_usd      numeric(24, 12),
    cache_write_usd     numeric(24, 12),
    reasoning_usd       numeric(24, 12),

    pricing_snapshot_id text,
    rates_applied       jsonb,
    context_tier        bigint,
    cost_warnings       text[]      NOT NULL DEFAULT '{}',

    project             text        NOT NULL DEFAULT 'unknown',
    feature             text,
    environment         text,
    subject_id          text,
    prompt_version      text,
    tags                jsonb       NOT NULL DEFAULT '{}'::jsonb,

    ingested_at         timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (ts, trace_id, span_id)
) PARTITION BY RANGE (ts);

CREATE INDEX IF NOT EXISTS usage_event_project_ts   ON usage_event (project, ts DESC);
CREATE INDEX IF NOT EXISTS usage_event_feature_ts   ON usage_event (feature, ts DESC);
CREATE INDEX IF NOT EXISTS usage_event_model_ts     ON usage_event (model_key, ts DESC);
CREATE INDEX IF NOT EXISTS usage_event_subject_ts   ON usage_event (subject_id, ts DESC);
CREATE INDEX IF NOT EXISTS usage_event_tags_gin     ON usage_event USING gin (tags);
-- Partial index: unpriced events are rare but we query them constantly for the banner.
CREATE INDEX IF NOT EXISTS usage_event_unpriced
    ON usage_event (ts DESC) WHERE cost_usd IS NULL;

CREATE TABLE IF NOT EXISTS usage_rollup_hourly (
    bucket              timestamptz NOT NULL,
    project             text        NOT NULL,
    feature             text        NOT NULL DEFAULT '',
    environment         text        NOT NULL DEFAULT '',
    model_key           text        NOT NULL DEFAULT '',
    provider            text        NOT NULL DEFAULT '',

    requests            bigint      NOT NULL DEFAULT 0,
    subjects            bigint      NOT NULL DEFAULT 0,
    input_tokens        bigint      NOT NULL DEFAULT 0,
    output_tokens       bigint      NOT NULL DEFAULT 0,
    cache_read_tokens   bigint      NOT NULL DEFAULT 0,
    cache_write_tokens  bigint      NOT NULL DEFAULT 0,
    cost_usd            numeric(24, 12) NOT NULL DEFAULT 0,
    unpriced_events     bigint      NOT NULL DEFAULT 0,
    refreshed_at        timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (bucket, project, feature, environment, model_key, provider)
);

CREATE INDEX IF NOT EXISTS usage_rollup_bucket ON usage_rollup_hourly (bucket DESC);

CREATE TABLE IF NOT EXISTS budget (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name         text        NOT NULL,
    scope        jsonb       NOT NULL DEFAULT '{}'::jsonb,
    amount_usd   numeric(18, 6) NOT NULL CHECK (amount_usd > 0),
    period       text        NOT NULL DEFAULT 'monthly'
                             CHECK (period IN ('monthly', 'rolling')),
    rolling_days integer     CHECK (rolling_days IS NULL OR rolling_days > 0),
    thresholds   numeric[]   NOT NULL DEFAULT '{0.5,0.8,1.0}',
    webhook_url  text,
    webhook_secret text,
    enabled      boolean     NOT NULL DEFAULT true,
    created_at   timestamptz NOT NULL DEFAULT now(),

    CHECK (period <> 'rolling' OR rolling_days IS NOT NULL)
);

-- One row per (budget, period, threshold): the UNIQUE constraint *is* the
-- fire-once state machine. Re-evaluating a budget cannot re-alert.
CREATE TABLE IF NOT EXISTS budget_alert (
    id           bigserial PRIMARY KEY,
    budget_id    uuid        NOT NULL REFERENCES budget(id) ON DELETE CASCADE,
    period_start date        NOT NULL,
    threshold    numeric     NOT NULL,
    spend_usd    numeric(24, 12) NOT NULL,
    amount_usd   numeric(18, 6)  NOT NULL,
    fired_at     timestamptz NOT NULL DEFAULT now(),
    delivered    boolean     NOT NULL DEFAULT false,
    delivery_error text,

    UNIQUE (budget_id, period_start, threshold)
);

CREATE TABLE IF NOT EXISTS anomaly (
    id             bigserial PRIMARY KEY,
    bucket         timestamptz NOT NULL,
    scope_key      text        NOT NULL DEFAULT 'global',
    observed_usd   numeric(24, 12) NOT NULL,
    baseline_usd   numeric(24, 12) NOT NULL,
    deviation_usd  numeric(24, 12) NOT NULL,
    score          numeric(12, 4)  NOT NULL,
    probable_cause jsonb       NOT NULL DEFAULT '{}'::jsonb,
    detected_at    timestamptz NOT NULL DEFAULT now(),

    UNIQUE (bucket, scope_key)
);

CREATE INDEX IF NOT EXISTS anomaly_bucket ON anomaly (bucket DESC);
