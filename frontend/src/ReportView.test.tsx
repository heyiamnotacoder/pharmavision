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
`;

const report: Report = {
  model_id: "muse-spark-1.3",
  markdown,
  drafts: [
    {
      audit: {
        numbers: [
          { text: "3", sentence: "Each group had 3 trials.", source: "len(trials)" },
          { text: "999", sentence: "The drug group travelled 999 cm.", source: null },
          { text: "110", sentence: "Vehicle mice travelled 110 cm.", source: "stats.mean" },
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
    expect(screen.getByText("[failed the number audit]")).toHaveClass("audit-flag");
    expect(screen.getByText("[TO FILL: how the sample size was decided]")).toHaveClass("to-fill");
    expect(screen.getByText(/Vehicle mice travelled 110 cm/).closest("del")).toBeNull();
    expect(screen.getByText("Each group had 3 trials.")).not.toHaveClass("to-fill");
  });

  it("says how the last draft did in the number audit and what is left to fill", () => {
    render(<ReportView report={report} />);
    expect(screen.getByRole("status")).toHaveTextContent(
      "Number audit failed: 1 of 3 numbers matched no computed value, so 1 sentence is struck out. 1 item to fill.",
    );
  });
});
