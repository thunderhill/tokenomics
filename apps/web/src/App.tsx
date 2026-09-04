import { useEffect, useMemo, useState } from "react";
import { AnomalyFeed } from "./components/AnomalyFeed";
import { BudgetGauges } from "./components/BudgetGauges";
import { type Metric, SpendChart } from "./components/SpendChart";
import { StatTiles } from "./components/StatTiles";
import { TokenBreakdown } from "./components/TokenBreakdown";
import { TopFeatures } from "./components/TopFeatures";
import {
  type AnomalyOut,
  type BudgetStatusOut,
  type ForecastOut,
  type SeriesPoint,
  type SpendRow,
  type TokenEconomicsRow,
  type UnpricedModel,
  api,
} from "./lib/api";
import { pct, tokens as fmtTokens, usd, usdPerM } from "./lib/format";

const PRESETS = [
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
] as const;

const METRICS = [
  { label: "cost", value: "cost" },
  { label: "tokens", value: "tokens" },
] as const satisfies readonly { label: string; value: Metric }[];

interface Snapshot {
  status: string;
  pricing_snapshot: string;
  models_priced: number;
}

interface Loaded {
  series: SeriesPoint[];
  breakdown: SpendRow[];
  features: SpendRow[];
  forecast: ForecastOut | null;
  anomalies: AnomalyOut[];
  unpriced: UnpricedModel[];
  budgets: BudgetStatusOut[];
  tokenTotals: TokenEconomicsRow | null;
  byModel: TokenEconomicsRow[];
}

const EMPTY: Loaded = {
  series: [],
  breakdown: [],
  features: [],
  forecast: null,
  anomalies: [],
  unpriced: [],
  budgets: [],
  tokenTotals: null,
  byModel: [],
};

function since(days: number): string {
  const date = new Date();
  date.setUTCDate(date.getUTCDate() - days);
  return date.toISOString();
}

export default function App() {
  const [days, setDays] = useState<number>(30);
  const [metric, setMetric] = useState<Metric>("cost");
  const [data, setData] = useState<Loaded>(EMPTY);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const range = useMemo(() => ({ since: since(days) }), [days]);

  useEffect(() => {
    api.health().then(setSnapshot).catch(() => undefined);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    async function load() {
      const [
        series,
        breakdown,
        features,
        forecast,
        anomalies,
        unpriced,
        budgetList,
        tokenTotals,
        byModel,
      ] = await Promise.all([
        api.spendSeries(range, days > 21 ? "day" : "hour", ["project"]),
        api.spendBreakdown(range, ["project"]),
        api.spendBreakdown(range, ["project", "feature"]),
        api.forecast(range),
        api.anomalies(range),
        api.unpriced(range),
        api.budgets(),
        api.tokenTotals(range),
        api.tokenEconomics(range, ["model_key"]),
      ]);
      const budgets = await Promise.all(budgetList.map((budget) => api.budgetStatus(budget.id)));
      if (cancelled) return;
      setData({
        series,
        breakdown,
        features: features.slice(0, 8),
        forecast,
        anomalies,
        unpriced,
        budgets,
        tokenTotals,
        byModel: byModel.slice(0, 8),
      });
    }

    load()
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [range, days]);

  const totalSpend = data.breakdown.reduce((sum, row) => sum + Number(row.cost_usd), 0);
  const unpricedEvents = data.breakdown.reduce((sum, row) => sum + row.unpriced_events, 0);
  const totals = data.tokenTotals;

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Tokenomics</h1>
          {snapshot && (
            <div className="snapshot">
              pricing snapshot {snapshot.pricing_snapshot.slice(0, 12)} - {snapshot.models_priced.toLocaleString()}{" "}
              models priced
            </div>
          )}
        </div>
        <div className="date-range" role="group" aria-label="Date range">
          {PRESETS.map((preset) => (
            <button
              key={preset.days}
              type="button"
              aria-pressed={days === preset.days}
              onClick={() => setDays(preset.days)}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </header>

      {error && (
        <div className="banner">
          <strong>Could not reach the API.</strong> {error}
        </div>
      )}

      {unpricedEvents > 0 && (
        <div className="banner">
          <strong>{unpricedEvents.toLocaleString()} event(s)</strong> could not be priced in this window --
          spend below is understated. See the table at the bottom for the models involved.
        </div>
      )}

      <StatTiles
        forecast={data.forecast}
        totalSpend={totalSpend}
        unpricedEvents={unpricedEvents}
        totals={totals}
      />

      <div className="grid-2">
        <div className="card">
          <div className="card-head">
            <div>
              <h2>{metric === "cost" ? "Spend" : "Tokens"} over time</h2>
              <div className="card-sub">
                by project, {days > 21 ? "daily" : "hourly"} buckets
              </div>
            </div>
            <div className="toggle" role="group" aria-label="Chart metric">
              {METRICS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  aria-pressed={metric === option.value}
                  onClick={() => setMetric(option.value)}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
          <SpendChart series={data.series} dimension="project" metric={metric} />
        </div>
        <div className="card">
          <h2>Budgets</h2>
          <div className="card-sub">current period burn-down</div>
          <BudgetGauges statuses={data.budgets} />
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h2>Top features by spend</h2>
          <div className="card-sub">project / feature, this window</div>
          <TopFeatures rows={data.features} />
        </div>
        <div className="card">
          <h2>Anomalies</h2>
          <div className="card-sub">deseasonalized vs. same hour-of-week baseline</div>
          <AnomalyFeed anomalies={data.anomalies} />
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h2>Where the money goes</h2>
          <div className="card-sub">
            share of token volume against share of spend, on the five disjoint buckets
            the bill is computed from
          </div>
          <TokenBreakdown row={totals} />
        </div>
        <div className="card">
          <h2>By model</h2>
          <div className="card-sub">effective blended rate, this window</div>
          {data.byModel.length === 0 ? (
            <div className="empty-state">No priced traffic in this window yet.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>model</th>
                  <th className="num">tokens</th>
                  <th className="num">spend</th>
                  <th className="num">$/1M</th>
                  <th className="num">cached</th>
                </tr>
              </thead>
              <tbody>
                {data.byModel.map((row) => (
                  <tr key={row.dimensions.model_key ?? "unknown"}>
                    <td>{row.dimensions.model_key || "unpriced"}</td>
                    <td className="num">{fmtTokens(row.total_tokens)}</td>
                    <td className="num">{usd(row.cost_usd)}</td>
                    <td className="num">{usdPerM(row.usd_per_1m_tokens)}</td>
                    <td className="num">
                      {row.cache_hit_rate === null ? "-" : pct(row.cache_hit_rate, 0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {data.unpriced.length > 0 && (
        <div className="card">
          <h2>Unpriced models</h2>
          <div className="card-sub">stored with a NULL cost, never zero -- these need a pricing update</div>
          <table>
            <thead>
              <tr>
                <th>model</th>
                <th>provider</th>
                <th className="num">events</th>
                <th className="num">tokens</th>
              </tr>
            </thead>
            <tbody>
              {data.unpriced.map((row) => (
                <tr key={`${row.model}-${row.provider}`}>
                  <td>{row.model ?? "unknown"}</td>
                  <td>{row.provider ?? "-"}</td>
                  <td className="num">{row.events.toLocaleString()}</td>
                  <td className="num">{fmtTokens(row.tokens)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {loading && <div className="empty-state">Loading...</div>}
    </div>
  );
}
