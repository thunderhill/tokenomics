import type { SpendRow } from "../lib/api";
import { compactNumber, pct, tokens, usd, usdPerM } from "../lib/format";

interface Props {
  rows: SpendRow[];
}

export function TopFeatures({ rows }: Props) {
  if (rows.length === 0) {
    return <div className="empty-state">No spend recorded for this window yet.</div>;
  }

  const total = rows.reduce((sum, row) => sum + Number(row.cost_usd), 0);

  return (
    <table>
      <thead>
        <tr>
          <th>project / feature</th>
          <th className="num">requests</th>
          <th className="num">tokens</th>
          <th className="num">spend</th>
          <th className="num">share</th>
          <th className="num">$/1M</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const cost = Number(row.cost_usd);
          // The reported totals are inclusive, so input + output *is* the whole volume.
          // Adding the cache or reasoning counts on top would double-count them.
          const totalTokens = row.input_tokens + row.output_tokens;
          const key = Object.values(row.dimensions).filter(Boolean).join(" / ");
          return (
            <tr key={key}>
              <td>{key || "unknown"}</td>
              <td className="num">{compactNumber(row.requests)}</td>
              <td className="num">{tokens(totalTokens)}</td>
              <td className="num">{usd(cost)}</td>
              <td className="num">{total ? pct(cost / total, 1) : "-"}</td>
              <td className="num">
                {totalTokens ? usdPerM((cost / totalTokens) * 1_000_000) : "-"}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
