import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SeriesPoint } from "../lib/api";
import { colorFor } from "../lib/colors";
import { shortDate, tokens, usd } from "../lib/format";

/**
 * Cost and volume are plotted by the same chart on purpose. Read alone, a rising cost
 * line is ambiguous -- more traffic and a shift to a pricier model look identical.
 * Flipping to volume separates them in one click: cost up while tokens are flat is a
 * model-mix change, not growth.
 */
export type Metric = "cost" | "tokens";

interface Props {
  series: SeriesPoint[];
  dimension: string;
  metric?: Metric;
}

const FORMAT: Record<Metric, (value: number) => string> = {
  cost: (value) => usd(value),
  tokens: (value) => `${tokens(value)} tok`,
};

const AXIS_FORMAT: Record<Metric, (value: number) => string> = {
  cost: (value) => usd(value, { maximumFractionDigits: 0 }),
  tokens: tokens,
};

function valueOf(point: SeriesPoint, metric: Metric): number {
  // Inclusive totals: input already contains the cache counts, output the reasoning.
  return metric === "cost"
    ? Number(point.cost_usd)
    : point.input_tokens + point.output_tokens;
}

interface Row {
  bucket: string;
  [project: string]: string | number;
}

function pivot(
  series: SeriesPoint[],
  dimension: string,
  metric: Metric,
): { rows: Row[]; keys: string[] } {
  const buckets = new Map<string, Row>();
  const keys: string[] = [];
  for (const point of series) {
    const key = point.dimensions[dimension] ?? "unknown";
    if (!keys.includes(key)) keys.push(key);
    const row = buckets.get(point.bucket) ?? { bucket: point.bucket };
    row[key] = valueOf(point, metric);
    buckets.set(point.bucket, row);
  }
  return { rows: Array.from(buckets.values()).sort((a, b) => a.bucket.localeCompare(b.bucket)), keys };
}

function ChartTooltip({
  active,
  payload,
  label,
  metric = "cost",
}: {
  active?: boolean;
  payload?: { dataKey: string; value: number; color: string }[];
  label?: string;
  metric?: Metric;
}) {
  if (!active || !payload?.length) return null;
  const total = payload.reduce((sum, entry) => sum + entry.value, 0);
  return (
    <div
      style={{
        background: "var(--surface-card)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: "8px 12px",
        fontSize: 12,
        boxShadow: "0 4px 16px rgba(0,0,0,0.12)",
      }}
    >
      <div style={{ color: "var(--text-muted)", marginBottom: 4 }}>{label && shortDate(label)}</div>
      {payload
        .slice()
        .sort((a, b) => b.value - a.value)
        .map((entry) => (
          <div key={entry.dataKey} style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span
                style={{ width: 8, height: 8, borderRadius: 2, background: entry.color, display: "inline-block" }}
              />
              {entry.dataKey}
            </span>
            <strong style={{ fontVariantNumeric: "tabular-nums" }}>
              {FORMAT[metric](entry.value)}
            </strong>
          </div>
        ))}
      <div
        style={{
          marginTop: 4,
          paddingTop: 4,
          borderTop: "1px solid var(--gridline)",
          display: "flex",
          justifyContent: "space-between",
          fontWeight: 600,
        }}
      >
        <span>total</span>
        <span>{FORMAT[metric](total)}</span>
      </div>
    </div>
  );
}

export function SpendChart({ series, dimension, metric = "cost" }: Props) {
  const { rows, keys } = pivot(series, dimension, metric);

  if (rows.length === 0) {
    return <div className="empty-state">Nothing recorded for this window yet.</div>;
  }

  return (
    <div>
      <div className="legend-row">
        {keys.map((key) => (
          <span key={key} className="legend-item">
            <span className="legend-swatch" style={{ background: colorFor(key) }} />
            {key}
          </span>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={rows} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="var(--gridline)" vertical={false} />
          <XAxis
            dataKey="bucket"
            tickFormatter={shortDate}
            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
            axisLine={{ stroke: "var(--baseline)" }}
            tickLine={false}
            minTickGap={32}
          />
          <YAxis
            tickFormatter={AXIS_FORMAT[metric]}
            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={64}
          />
          <Tooltip content={<ChartTooltip metric={metric} />} />
          {keys.map((key) => (
            <Area
              key={key}
              type="monotone"
              dataKey={key}
              stackId={metric}
              stroke={colorFor(key)}
              strokeWidth={2}
              fill={colorFor(key)}
              fillOpacity={0.18}
              // The mount-in clip animation never advances in a static render (SSR,
              // headless screenshot, PDF export), which leaves the chart permanently
              // blank. A dashboard that refetches on every date-range change would
              // also replay this as a distracting slide-in each time.
              isAnimationActive={false}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
