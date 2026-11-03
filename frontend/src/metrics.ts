import type { Metrics } from "./api";

export interface MetricRow {
  label: string;
  /** Formatted; null for a metric the measure stage could not compute (never shown as 0). */
  value: string | null;
  unit: string;
  /** The measure settings that produced the value (`manifest.measure.version`). */
  config: string;
}

/**
 * Display order, labels, units and decimals. The units copy `METRIC_UNITS` in `artifacts.py`
 * by hand: the API does not serve them yet, so a new or renamed metric changes both.
 */
const METRICS: readonly {
  key: keyof Metrics;
  label: string;
  unit: string;
  decimals: number;
}[] = [
  { key: "distance_travelled_cm", label: "Distance travelled", unit: "cm", decimals: 1 },
  { key: "mean_velocity_cm_s", label: "Mean velocity", unit: "cm/s", decimals: 2 },
  { key: "centre_time_s", label: "Centre time", unit: "s", decimals: 1 },
  { key: "centre_entries", label: "Centre entries", unit: "entries", decimals: 0 },
  { key: "thigmotaxis", label: "Thigmotaxis", unit: "fraction", decimals: 2 },
  { key: "immobility_s", label: "Immobility", unit: "s", decimals: 1 },
];

export function metricRows(metrics: Metrics, measureVersion: number): MetricRow[] {
  return METRICS.map(({ key, label, unit, decimals }) => {
    const value = metrics[key];
    return {
      label,
      value: value === null ? null : value.toFixed(decimals),
      unit,
      config: `measure v${String(measureVersion)}`,
    };
  });
}
