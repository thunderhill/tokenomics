import type { ForecastOut, TokenEconomicsRow } from "../lib/api";
import { pct, tokens, usd, usdPerM, usdSigned } from "../lib/format";

interface Props {
  forecast: ForecastOut | null;
  totalSpend: number;
  unpricedEvents: number;
  totals: TokenEconomicsRow | null;
}

export function StatTiles({ forecast, totalSpend, unpricedEvents, totals }: Props) {
  const netCache = totals ? Number(totals.net_cache_benefit_usd) : 0;

  return (
    <div className="stat-row">
      <div className="stat-tile">
        <div className="label">Spend (window)</div>
        <div className="value">{usd(totalSpend)}</div>
        {forecast && <div className="sub">{usd(forecast.daily_run_rate_usd)} / day</div>}
      </div>

      <div className="stat-tile">
        <div className="label">Tokens (window)</div>
        <div className="value">{totals ? tokens(totals.total_tokens) : "-"}</div>
        {totals && (
          <div className="sub">
            {tokens(totals.input_tokens)} in / {tokens(totals.output_tokens)} out
          </div>
        )}
      </div>

      <div className="stat-tile">
        <div className="label">Blended rate</div>
        <div className="value">{totals ? usdPerM(totals.usd_per_1m_tokens) : "-"}</div>
        <div className="sub">across every model in this window</div>
      </div>

      <div className="stat-tile">
        <div className="label">Cache benefit</div>
        <div
          className="value"
          style={{ color: netCache < 0 ? "var(--status-critical)" : undefined }}
        >
          {totals ? usdSigned(netCache) : "-"}
        </div>
        {totals && (
          <div className="sub">
            {/* Reads save, writes cost a premium. Netting them is the only honest
                version: a workload that rewrites its cache on every call looks
                excellent on savings alone. */}
            {usd(totals.cache_savings_usd)} saved &minus; {usd(totals.cache_write_premium_usd)}{" "}
            written
            {totals.cache_hit_rate !== null && ` · ${pct(totals.cache_hit_rate)} hit rate`}
          </div>
        )}
      </div>

      <div className="stat-tile">
        <div className="label">Projected month-end</div>
        <div className="value">{forecast ? usd(forecast.projected_month_end_usd) : "-"}</div>
        {forecast && (
          <div className="sub">
            {usd(forecast.lower_usd)} - {usd(forecast.upper_usd)} ({forecast.method})
          </div>
        )}
      </div>

      <div className="stat-tile">
        <div className="label">Unpriced events</div>
        <div
          className="value"
          style={{ color: unpricedEvents > 0 ? "var(--status-warning)" : undefined }}
        >
          {unpricedEvents}
        </div>
        <div className="sub">stored as NULL, never $0</div>
      </div>
    </div>
  );
}
