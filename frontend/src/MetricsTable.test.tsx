import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MetricsTable } from "./MetricsTable";

describe("MetricsTable", () => {
  it("shows each metric with its value, unit and configuration version", () => {
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
    expect(within(table).getAllByRole("columnheader").map((th) => th.textContent)).toEqual([
      "Metric",
      "Value",
      "Unit",
      "Configuration",
    ]);
    const distance = screen.getByRole("row", { name: /Distance travelled/ });
    expect(within(distance).getAllByRole("cell").map((td) => td.textContent)).toEqual([
      "Distance travelled",
      "1200.0",
      "cm",
      "measure v4",
    ]);
    const immobility = screen.getByRole("row", { name: /Immobility/ });
    expect(within(immobility).getByText("not computed")).toBeInTheDocument();
  });
});
