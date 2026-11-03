import { describe, expect, it } from "vitest";
import type { Metrics } from "./api";
import { metricRows } from "./metrics";

const metrics: Metrics = {
  distance_travelled_cm: 3456.789,
  mean_velocity_cm_s: 5.7613,
  centre_time_s: 42.35,
  centre_entries: 17,
  thigmotaxis: 0.8123,
  immobility_s: null,
};

describe("metricRows", () => {
  it("gives every open-field metric its unit and the measure settings version", () => {
    expect(metricRows(metrics, 4)).toEqual([
      { label: "Distance travelled", value: "3456.8", unit: "cm", config: "measure v4" },
      { label: "Mean velocity", value: "5.76", unit: "cm/s", config: "measure v4" },
      { label: "Centre time", value: "42.4", unit: "s", config: "measure v4" },
      { label: "Centre entries", value: "17", unit: "entries", config: "measure v4" },
      { label: "Thigmotaxis", value: "0.81", unit: "fraction", config: "measure v4" },
      { label: "Immobility", value: null, unit: "s", config: "measure v4" },
    ]);
  });
});
