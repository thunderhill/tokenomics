import type { BudgetStatusOut } from "../lib/api";
import { pct, usd } from "../lib/format";

interface Props {
  statuses: BudgetStatusOut[];
}

function gaugeColor(utilization: number): string {
  if (utilization >= 1) return "var(--status-critical)";
  if (utilization >= 0.8) return "var(--status-warning)";
  return "var(--status-good)";
}

export function BudgetGauges({ statuses }: Props) {
  if (statuses.length === 0) {
    return <div className="empty-state">No budgets configured yet.</div>;
  }

  return (
    <div>
      {statuses.map((status) => {
        const width = Math.min(status.utilization, 1) * 100;
        const color = gaugeColor(status.utilization);
        return (
          <div className="budget" key={status.budget.id}>
            <div className="budget-head">
              <span className="name">{status.budget.name}</span>
              <span className="amounts">
                {usd(status.spend_usd)} / {usd(status.amount_usd)} ({pct(status.utilization, 0)})
              </span>
            </div>
            <div className="gauge-track">
              <div className="gauge-fill" style={{ width: `${width}%`, background: color }} />
            </div>
            {status.has_blind_spot && (
              <div style={{ color: "var(--status-warning)", fontSize: 11, marginTop: 4 }}>
                {status.unpriced_events} unpriced event(s) in scope -- actual spend is higher
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
