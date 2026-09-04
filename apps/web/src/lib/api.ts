// Hand-written types matching the FastAPI response models in src/tokenomics/api/routers.
// A generated OpenAPI client was the original plan; a hand client is a deliberate
// scope trade for v0.1 -- it is a fraction of the surface and does not drift silently,
// because every field here is read by a component below.

const BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "";

export interface CostComponents {
  input_usd: string;
  output_usd: string;
  cache_read_usd: string;
  cache_write_usd: string;
  reasoning_usd: string;
}

export interface SpendRow {
  dimensions: Record<string, string | null>;
  requests: number;
  subjects: number;
  cost_usd: string;
  // Reported totals, in semconv form: input_tokens already contains the cache counts
  // and output_tokens already contains reasoning. Never add the components to these.
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  reasoning_tokens: number;
  billable_input_tokens: number;
  cost_components: CostComponents;
  unpriced_events: number;
}

/** The five disjoint buckets the bill is actually computed from. */
export type Component = "cache_read" | "cache_write" | "input" | "output" | "reasoning";

export const COMPONENTS: Component[] = [
  "cache_read",
  "cache_write",
  "input",
  "output",
  "reasoning",
];

export const COMPONENT_LABELS: Record<Component, string> = {
  cache_read: "cache read",
  cache_write: "cache write",
  input: "input",
  output: "output",
  reasoning: "reasoning",
};

export interface TokenEconomicsRow extends SpendRow {
  total_tokens: number;
  token_components: Record<Component, number>;
  token_shares: Record<Component, number | null>;
  cost_shares: Record<Component, number | null>;
  usd_per_1m_tokens: string | null;
  usd_per_1m_by_component: Record<Component, string | null>;
  cache_hit_rate: number | null;
  reasoning_share: number | null;
  cache_savings_usd: string;
  cache_write_premium_usd: string;
  net_cache_benefit_usd: string;
  cache_priced_coverage: number | null;
}

export interface SeriesPoint extends SpendRow {
  bucket: string;
}

export interface UnpricedModel {
  model: string | null;
  provider: string | null;
  events: number;
  tokens: number;
}

export interface ForecastOut {
  month_to_date_usd: string;
  projected_month_end_usd: string;
  lower_usd: string;
  upper_usd: string;
  daily_run_rate_usd: string;
  days_observed: number;
  days_remaining: number;
  method: string;
  day_of_week_factors: number[];
}

export interface Cause {
  dimension: string;
  value: string;
  delta_usd: string;
  share: number;
}

export interface AnomalyOut {
  bucket: string;
  observed_usd: string;
  baseline_usd: string;
  deviation_usd: string;
  score: number;
  multiple: number;
  probable_cause: Cause[];
}

export interface BudgetOut {
  id: string;
  name: string;
  amount_usd: string;
  scope: Record<string, unknown>;
  period: "monthly" | "rolling";
  rolling_days: number | null;
  thresholds: string[];
  webhook_url: string | null;
  has_webhook_secret: boolean;
  enabled: boolean;
}

export interface BudgetStatusOut {
  budget: BudgetOut;
  period_start: string;
  since: string;
  until: string;
  spend_usd: string;
  amount_usd: string;
  remaining_usd: string;
  utilization: number;
  requests: number;
  unpriced_events: number;
  has_blind_spot: boolean;
  thresholds_crossed: string[];
}

class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(`API ${status}: ${detail}`);
  }
}

async function get<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
  const url = new URL(path, BASE_URL || window.location.origin);
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined) url.searchParams.append(key, String(value));
  }
  const response = await fetch(url.toString().replace(window.location.origin, ""));
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(response.status, body.detail ?? response.statusText);
  }
  return response.json() as Promise<T>;
}

export interface RangeParams {
  since: string;
  until?: string;
  [key: string]: string | undefined;
}

export const api = {
  spendBreakdown: (range: RangeParams, groupBy: string[] = ["project"]) =>
    get<SpendRow[]>("/api/spend/breakdown", { ...range, group_by: groupBy.join(",") }),

  spendSeries: (range: RangeParams, granularity: "hour" | "day" | "week" | "month", groupBy?: string[]) =>
    get<SeriesPoint[]>("/api/spend/series", {
      ...range,
      granularity,
      group_by: groupBy?.join(","),
    }),

  tokenEconomics: (range: RangeParams, groupBy: string[] = ["project"]) =>
    get<TokenEconomicsRow[]>("/api/spend/tokens", { ...range, group_by: groupBy.join(",") }),

  // One ungrouped row for the window. Folding the grouped rows client-side would
  // weight ratios by slice rather than by volume, and double-count subjects.
  tokenTotals: (range: RangeParams) =>
    get<TokenEconomicsRow>("/api/spend/tokens/summary", range),

  unpriced: (range: RangeParams) => get<UnpricedModel[]>("/api/spend/unpriced", range),

  forecast: (range: RangeParams) => get<ForecastOut>("/api/forecast", range),

  anomalies: (range: RangeParams) => get<AnomalyOut[]>("/api/anomalies", range),

  budgets: () => get<BudgetOut[]>("/api/budgets"),

  budgetStatus: (id: string) => get<BudgetStatusOut>(`/api/budgets/${id}/status`),

  health: () => get<{ status: string; pricing_snapshot: string; models_priced: number }>("/health"),
};

export { ApiError };
