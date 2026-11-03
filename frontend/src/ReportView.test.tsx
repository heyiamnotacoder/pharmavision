import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Report } from "./api";
import { ReportView } from "./ReportView";

// As the report stage ships it: `arrive.assemble`'s placeholders, `arrive.strike`'s strike.
const markdown = `# Methods and results

## Sample size

Each group had 3 trials.

[TO FILL: how the sample size was decided]

## Results

~~The drug group travelled 999 cm.~~ **[failed the number audit]** Vehicle mice travelled 110 cm.

| Group | Distance (cm) |
|---|---|
| ~~drug~~ | ~~999~~ **[failed the number audit]** |
`;

const report: Report = {
  model_id: "muse-spark-1.3",
  markdown,
  drafts: [
    {
      audit: {
        numbers: [
          {
            text: "3",
            sentence: "Each group had 3 trials.",
            sentence_span: [52, 76],
            source: "len(trials)",
          },
          {
            text: "999",
            sentence: "The drug group travelled 999 cm.",
            sentence_span: [144, 176],
            source: null,
          },
          {
            text: "110",
            sentence: "Vehicle mice travelled 110 cm.",
            sentence_span: [177, 207],
            source: "stats.mean",
          },
          {
            text: "999",
            sentence: "| drug | 999 |",
            sentence_span: [240, 254],
            source: null,
          },
        ],
      },
    },
  ],
};

describe("ReportView", () => {
  it("marks struck-out sentences and [TO FILL] items apart from the audited prose", () => {
    render(<ReportView report={report} />);
    const struck = screen.getByText("The drug group travelled 999 cm.");
    expect(struck.tagName).toBe("DEL");
    expect(struck).toHaveClass("struck");
    for (const flag of screen.getAllByText("[failed the number audit]")) {
      expect(flag).toHaveClass("audit-flag");
    }
    expect(screen.getByText("[TO FILL: how the sample size was decided]")).toHaveClass("to-fill");
    expect(screen.getByText(/Vehicle mice travelled 110 cm/).closest("del")).toBeNull();
    expect(screen.getByText("Each group had 3 trials.")).not.toHaveClass("to-fill");
    // A struck table row, as `arrive.strike` writes it: each cell struck, still a table row.
    const row = screen.getByRole("row", { name: /drug/ });
    expect(row.querySelectorAll("td del.struck")).toHaveLength(2);
    expect(row).toHaveTextContent("[failed the number audit]");
  });

  it("says how the last draft did in the number audit and what is left to fill", () => {
    render(<ReportView report={report} />);
    expect(screen.getByRole("status")).toHaveTextContent(
      "Number audit failed: 2 of 4 numbers matched no computed value, so 2 sentences are struck out. 1 item to fill.",
    );
  });
});
