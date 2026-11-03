import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Arena, Events, Manifest, RunView } from "./api";
import { RunViewer } from "./RunViewer";

const run: RunView = {
  run_id: "r1",
  status: "done",
  stage: null,
  error: null,
  read_only: true,
  stages_done: ["ingest", "arena", "track", "measure", "behaviours"],
  artifacts: {
    manifest: "/runs/r1/artifacts/manifest",
    video: "/runs/r1/video",
    events: "/runs/r1/artifacts/events",
  },
};
const manifest: Manifest = {
  template: { centre_area_fraction: 0.5 },
  measure: { version: 4 },
  video: { width: 64, height: 48, fps: 15, n_frames: 9000 },
};
const events: Events = {
  events: [
    {
      first_frame: 120,
      last_frame: 135,
      label: "rearing",
      confidence: 0.9,
      low_confidence: false,
      rationale: "up on its hind legs",
      source: "model",
    },
  ],
};
const arena: Arena = {
  floor_px: [
    [4, 4],
    [60, 4],
    [60, 44],
    [4, 44],
  ],
  centre_px: [],
  px_per_cm: 1.2,
  confirmed: false,
};
const atGate: RunView = {
  ...run,
  run_id: "r2",
  status: "awaiting zone confirmation",
  read_only: false,
  stages_done: ["ingest", "arena"],
  artifacts: {
    manifest: "/runs/r2/artifacts/manifest",
    video: "/runs/r2/video",
    background: "/runs/r2/background",
    arena: "/runs/r2/artifacts/arena",
  },
};
let served: Record<string, unknown>;

beforeEach(() => {
  served = {
    "/api/runs/r1": run,
    "/api/runs/r1/artifacts/manifest": manifest,
    "/api/runs/r1/artifacts/events": events,
    "/api/runs/r2": atGate,
    "/api/runs/r2/artifacts/manifest": manifest,
    "/api/runs/r2/artifacts/arena": arena,
    "/api/runs/r2/arena/confirm": { ...atGate, status: "queued" },
  };
  vi.stubGlobal("fetch", (url: string) =>
    Promise.resolve(
      url in served
        ? new Response(JSON.stringify(served[url]))
        : new Response(null, { status: 404 }),
    ),
  );
  // jsdom has no video frame callbacks; the overlay's draw loop only needs them to exist.
  HTMLVideoElement.prototype.requestVideoFrameCallback = () => 0;
  HTMLVideoElement.prototype.cancelVideoFrameCallback = () => undefined;
  // jsdom has no 2D canvas: every drawing call does nothing.
  const noop = () => undefined;
  const ctx = new Proxy({}, { get: (_, key) => (key === "then" ? undefined : noop) });
  HTMLCanvasElement.prototype.getContext = (() =>
    ctx) as unknown as HTMLCanvasElement["getContext"];
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("RunViewer", () => {
  it("seeks the video into an event's first frame when the event is clicked", async () => {
    const { container } = render(<RunViewer runId="r1" />);
    fireEvent.click(await screen.findByRole("button", { name: /^rearing, frames 120–135/ }));
    const video = container.querySelector("video");
    expect(video?.currentTime).toBeCloseTo(120.5 / 15);
  });

  it("has a live run at the gate checked in the zone editor, then shows it running", async () => {
    render(<RunViewer runId="r2" />);
    const confirm = await screen.findByRole("button", { name: "Confirm zones" });
    // The API writes the confirmed arena before it answers.
    served["/api/runs/r2"] = { ...atGate, status: "running", stage: "track" };
    served["/api/runs/r2/artifacts/arena"] = { ...arena, confirmed: true };

    fireEvent.click(confirm);

    expect(await screen.findByText(/Zones confirmed/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Confirm zones" })).not.toBeInTheDocument();
  });

  it("shows a precomputed run at the gate without the zone editor", async () => {
    served["/api/runs/r2"] = { ...atGate, read_only: true };
    render(<RunViewer runId="r2" />);
    expect(await screen.findByText(/zones not confirmed/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Confirm zones" })).not.toBeInTheDocument();
  });
});
