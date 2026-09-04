import {
  COMPONENTS,
  COMPONENT_LABELS,
  type Component,
  type TokenEconomicsRow,
} from "../lib/api";
import { componentColor } from "../lib/colors";
import { pct, tokens, usd, usdPerM } from "../lib/format";

interface Props {
  row: TokenEconomicsRow | null;
}

interface Bar {
  label: string;
  shares: Record<Component, number | null>;
  title: (component: Component) => string;
}

/**
 * Volume above, money below, on the same five buckets and the same scale.
 *
 * The gap between the two bars is the entire point of the card: output is routinely a
 * tenth of the tokens and half the bill, and cache reads the reverse. A single bar --
 * either one alone -- hides that, which is how a workload gets optimized for the wrong
 * number.
 */
export function TokenBreakdown({ row }: Props) {
  if (!row || row.total_tokens === 0) {
    return <div className="empty-state">No token usage recorded for this window yet.</div>;
  }

  const bundledReasoning = row.reasoning_tokens - row.token_components.reasoning;

  const bars: Bar[] = [
    {
      label: "volume",
      shares: row.token_shares,
      title: (c) => `${COMPONENT_LABELS[c]}: ${tokens(row.token_components[c])} tokens`,
    },
    {
      label: "spend",
      shares: row.cost_shares,
      title: (c) => `${COMPONENT_LABELS[c]}: ${usd(row.cost_components[`${c}_usd`])}`,
    },
  ];

  return (
    <div>
      {bars.map((bar) => (
        <div key={bar.label} className="component-bar-row">
          <div className="component-bar-label">{bar.label}</div>
          <div className="component-bar">
            {COMPONENTS.map((component) => {
              const share = bar.shares[component] ?? 0;
              if (share <= 0) return null;
              return (
                <div
                  key={component}
                  className="component-bar-segment"
                  style={{ width: `${share * 100}%`, background: componentColor(component) }}
                  title={`${bar.title(component)} (${pct(share)})`}
                />
              );
            })}
          </div>
        </div>
      ))}

      <table className="component-table">
        <thead>
          <tr>
            <th>component</th>
            <th className="num">tokens</th>
            <th className="num">of volume</th>
            <th className="num">spend</th>
            <th className="num">of spend</th>
            <th className="num">effective rate</th>
          </tr>
        </thead>
        <tbody>
          {COMPONENTS.map((component) => (
            <tr key={component}>
              <td>
                <span
                  className="legend-swatch"
                  style={{ background: componentColor(component) }}
                />
                {COMPONENT_LABELS[component]}
              </td>
              <td className="num">{tokens(row.token_components[component])}</td>
              <td className="num">
                {row.token_shares[component] === null ? "-" : pct(row.token_shares[component]!)}
              </td>
              <td className="num">{usd(row.cost_components[`${component}_usd`])}</td>
              <td className="num">
                {row.cost_shares[component] === null ? "-" : pct(row.cost_shares[component]!)}
              </td>
              <td className="num">{usdPerM(row.usd_per_1m_by_component[component])}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {bundledReasoning > 0 && (
        /* Without this the card contradicts itself: a project can show a large
           reasoning share while the reasoning row reads zero. Only ~58 models price
           reasoning apart; for the rest it is already inside the output charge. */
        <p className="card-note">
          {tokens(bundledReasoning)} reasoning tokens are reported but have no row above
          &mdash; these models quote no separate reasoning rate, so they are already
          inside the output charge.
        </p>
      )}
    </div>
  );
}
