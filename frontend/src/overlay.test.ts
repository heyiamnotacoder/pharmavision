import { describe, expect, it } from "vitest";
import { pathTail } from "./overlay";

const p = (x: number) => ({ x, y: 0 });

describe("pathTail", () => {
  it("holds the positions of the last tail frames up to the current one, oldest first", () => {
    const positions = [p(0), p(1), p(2), p(3), p(4), p(5)];
    const lines = pathTail(positions, 4, 3);
    expect(lines.map((line) => line.map((pt) => pt.x))).toEqual([[2, 3, 4]]);
  });

  it("ages points from 1 (oldest kept) to 0 (now)", () => {
    const positions = [p(0), p(1), p(2), p(3), p(4)];
    const [line] = pathTail(positions, 4, 3);
    expect(line?.map((pt) => pt.age)).toEqual([1, 0.5, 0]);
  });

  it("breaks the line at a gap instead of bridging it", () => {
    const positions = [p(0), p(1), null, p(3), p(4)];
    const lines = pathTail(positions, 4, 5);
    expect(lines.map((line) => line.map((pt) => pt.x))).toEqual([
      [0, 1],
      [3, 4],
    ]);
  });

  it("never shows positions after the current frame", () => {
    const positions = [p(0), p(1), p(2), p(3)];
    expect(pathTail(positions, 0, 10).map((line) => line.map((pt) => pt.x))).toEqual([[0]]);
  });
});
