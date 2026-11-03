import { describe, expect, it } from "vitest";
import { frameAt, seekTime } from "./frames";

describe("frameAt", () => {
  it("maps a frame's presentation time to its index at 15 fps", () => {
    expect(frameAt(0, 15, 9000)).toBe(0);
    expect(frameAt(1, 15, 9000)).toBe(15);
    expect(frameAt(7 / 15, 15, 9000)).toBe(7); // 7/15 * 15 is 6.999… in floating point
  });

  it("holds a frame until the next one starts", () => {
    expect(frameAt(0.1, 15, 9000)).toBe(1); // frame 1 spans 0.0667–0.1333 s
  });

  it("clamps to the frames the track has", () => {
    expect(frameAt(-0.5, 15, 9000)).toBe(0);
    expect(frameAt(601, 15, 9000)).toBe(8999);
  });
});

describe("seekTime", () => {
  it("lands inside the frame, so the video shows that frame and not the one before", () => {
    for (const frame of [0, 1, 7, 120, 8999]) {
      expect(frameAt(seekTime(frame, 15), 15, 9000)).toBe(frame);
    }
    expect(seekTime(15, 15)).toBeCloseTo(1 + 0.5 / 15); // the middle of frame 15
  });
});
