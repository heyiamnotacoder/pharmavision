import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { BehaviourEvent, Exclusion } from "./api";
import { Timeline } from "./Timeline";

const event = (first: number, last: number, label: BehaviourEvent["label"]): BehaviourEvent => ({
  first_frame: first,
  last_frame: last,
  label,
  confidence: 0.8,
  low_confidence: false,
  rationale: "forepaws off the floor",
  source: "model",
});

const exclusion: Exclusion = {
  scope: "window",
  first_frame: 3000,
  last_frame: 3090,
  reason: "a hand in the arena",
  source: "model",
};

describe("Timeline", () => {
  it("seeks to an event's first frame when it is clicked, and to an exclusion's", () => {
    const onSeek = vi.fn();
    render(
      <Timeline
        events={[event(120, 135, "rearing"), event(900, 910, "none")]}
        exclusions={[exclusion]}
        nFrames={9000}
        fps={15}
        frame={0}
        onSeek={onSeek}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /^none, frames 900–910/ }));
    expect(onSeek).toHaveBeenLastCalledWith(900);
    fireEvent.click(screen.getByRole("button", { name: /^excluded: window, frames 3000–3090/ }));
    expect(onSeek).toHaveBeenLastCalledWith(3000);
  });

  it("draws a trial exclusion apart from a window, and says it covers the whole trial", () => {
    render(
      <Timeline
        events={[]}
        exclusions={[{ ...exclusion, scope: "trial", first_frame: 0, last_frame: 8999 }]}
        nFrames={9000}
        fps={15}
        frame={0}
        onSeek={vi.fn()}
      />,
    );
    const bar = screen.getByRole("button", { name: /^excluded: whole trial, frames 0–8999/ });
    expect(bar).toHaveClass("exclusion", "trial");
  });

  it("says which lanes have no artifact yet instead of drawing them empty", () => {
    render(
      <Timeline
        events={null}
        exclusions={null}
        nFrames={9000}
        fps={15}
        frame={0}
        onSeek={vi.fn()}
      />,
    );
    expect(screen.getByText(/not judged yet/)).toBeInTheDocument();
    expect(screen.getByText(/no QC yet/)).toBeInTheDocument();
    expect(screen.queryAllByRole("button")).toHaveLength(0);
  });
});
