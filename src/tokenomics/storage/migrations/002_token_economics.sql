-- Carry token components and per-component cost through to the rollup table.
--
-- usage_event has stored reasoning_tokens and all five *_usd component columns since
-- 001, but the hourly rollup summed only the four token counts and the total. Any read
-- served from the rollup therefore could not answer "what share of the bill is output
-- tokens?" without falling back to the events table, and the two would disagree about
-- what they contained. Additive columns only: existing rows keep their values and are
-- refilled on the next `tokenomics db rollup`, which recomputes rather than increments.

ALTER TABLE usage_rollup_hourly
    ADD COLUMN IF NOT EXISTS reasoning_tokens       bigint NOT NULL DEFAULT 0,
    -- input minus what was served from (or written to) cache: the bucket actually
    -- billed at the full input rate. Stored rather than derived so the rollup does not
    -- have to re-apply the clamp on every read.
    ADD COLUMN IF NOT EXISTS billable_input_tokens  bigint NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS input_usd              numeric(24, 12) NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS output_usd             numeric(24, 12) NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS cache_read_usd         numeric(24, 12) NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS cache_write_usd        numeric(24, 12) NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS reasoning_usd          numeric(24, 12) NOT NULL DEFAULT 0;
