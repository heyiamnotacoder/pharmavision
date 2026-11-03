import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Qc } from "./api";
import { QcList } from "./QcList";

const qc: Qc = {
  flags: [
    {
      kind: "tracking_loss",
      first_frame: 0,
      last_frame: 8999,
      severity: "critical",
      reason: "tracking lost on 8% of frames, over the 5% limit",
    },
    {
      kind: "second_mask",
      first_frame: 3000,
      last_frame: 3090,
      severity: "warning",
      reason: "a second mask beside the animal",
    },
  ],
  exclusions: [
    {
      scope: "trial",
      first_frame: 0,
      last_frame: 8999,
      reason: "tracking lost on 8% of frames, over the 5% limit",
      source: "rule",
    },
    {
      scope: "window",
      first_frame: 3000,
      last_frame: 3090,
      reason: "a gloved hand moves the animal",
      source: "model",
    },
  ],
  reviews: [],
};

describe("QcList", () => {
  it("lists every exclusion with its frames, its reason and who decided it", () => {
    render(<QcList qc={qc} fps={15} onSeek={vi.fn()} />);
    const rows = within(screen.getByRole("table", { name: "QC exclusions" })).getAllByRole("row");
    expect(rows).toHaveLength(1 + qc.exclusions.length);
    expect(rows[1]).toHaveTextContent(/whole trial.*frames 0–8999.*over the 5% limit.*rule/);
    expect(rows[2]).toHaveTextContent(/window.*frames 3000–3090.*gloved hand.*model/);
  });

  it("says so when nothing was excluded", () => {
    render(<QcList qc={{ ...qc, exclusions: [] }} fps={15} onSeek={vi.fn()} />);
    expect(screen.queryByRole("table")).toBeNull();
    expect(screen.getByText(/No exclusions/)).toBeInTheDocument();
  });
});
