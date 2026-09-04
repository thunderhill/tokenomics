import type { AnomalyOut } from "../lib/api";
import { shortDateTime, usd } from "../lib/format";

interface Props {
  anomalies: AnomalyOut[];
}

export function AnomalyFeed({ anomalies }: Props) {
  if (anomalies.length === 0) {
    return <div className="empty-state">No anomalies in this window.</div>;
  }

  return (
    <div className="anomaly-list">
      {anomalies.map((item) => (
        <div className="anomaly-item" key={item.bucket}>
          <div className="headline">
            <span>{shortDateTime(item.bucket)}</span>
            <span>
              {usd(item.observed_usd)} vs {usd(item.baseline_usd)}{" "}
              <span className="multiple">{item.multiple.toFixed(1)}x</span>
            </span>
          </div>
          {item.probable_cause.length > 0 && (
            <div className="cause">
              probable cause:{" "}
              {item.probable_cause
                .map((cause) => `${cause.dimension}=${cause.value} (${Math.round(cause.share * 100)}%)`)
                .join(", ")}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
