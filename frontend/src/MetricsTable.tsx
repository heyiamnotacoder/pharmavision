import type { Metrics } from "./api";
import { metricRows } from "./metrics";

export function MetricsTable({
  metrics,
  measureVersion,
}: {
  metrics: Metrics;
  measureVersion: number;
}) {
  return (
    <table className="metrics" aria-label="Open field metrics">
      <thead>
        <tr>
          <th scope="col">Metric</th>
          <th scope="col">Value</th>
          <th scope="col">Unit</th>
          <th scope="col">Configuration</th>
        </tr>
      </thead>
      <tbody>
        {metricRows(metrics, measureVersion).map((row) => (
          <tr key={row.label}>
            <td>{row.label}</td>
            <td className={row.value === "not computed" ? "missing" : "number"}>{row.value}</td>
            <td>{row.unit}</td>
            <td>{row.config}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
