import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Events, Manifest, RunView } from "./api";
import { RunViewer } from "./RunViewer";

const run: RunView = {
  run_id: "r1",
  status: "done",
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
const served: Record<string, unknown> = {
  "/api/runs/r1": run,
  "/api/runs/r1/artifacts/manifest": manifest,
  "/api/runs/r1/artifacts/events": events,
};

beforeEach(() => {
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
  HTMLCanvasElement.prototype.getContext = (() => ({
    clearRect: () => undefined,
  })) as unknown as HTMLCanvasElement["getContext"];
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
});
