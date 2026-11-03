import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MetricsTable } from "./MetricsTable";

// Values, units and versions are `metricRows`' job (metrics.test.ts); this covers the table.
describe("MetricsTable", () => {
  it("lays each metric out as a row under the four headers, marking a missing value", () => {
    render(
      <MetricsTable
        metrics={{
          distance_travelled_cm: 1200,
          mean_velocity_cm_s: 2,
          centre_time_s: 30,
          centre_entries: 5,
          thigmotaxis: 0.5,
          immobility_s: null,
        }}
        measureVersion={4}
      />,
    );
    const table = screen.getByRole("table", { name: "Open field metrics" });
    expect(
      within(table)
        .getAllByRole("columnheader")
        .map((th) => th.textContent),
    ).toEqual(["Metric", "Value", "Unit", "Configuration"]);
    expect(within(table).getAllByRole("row")).toHaveLength(7);
    const immobility = screen.getByRole("row", { name: /Immobility/ });
    expect(within(immobility).getByText("not computed")).toHaveClass("missing");
  });
});
