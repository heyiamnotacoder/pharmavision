import { describe, expect, it } from "vitest";
import type { Point } from "./api";
import { centreZone, moveCorner, toImagePoint } from "./zones";

const square: Point[] = [
  [30, 30],
  [330, 30],
  [330, 330],
  [30, 330],
];

describe("centreZone", () => {
  it("is the floor shrunk about its centre to the template's fraction of its area", () => {
    // Half of a 300 px square: a 212.13 px square (300 / √2) about (180, 180).
    const half = 300 / Math.SQRT2 / 2;
    const centre = centreZone(square, 0.5);
    const expected = [
      [180 - half, 180 - half],
      [180 + half, 180 - half],
      [180 + half, 180 + half],
      [180 - half, 180 + half],
    ];
    centre.forEach(([x, y], i) => {
      expect(x).toBeCloseTo(expected[i]?.[0] ?? NaN, 6);
      expect(y).toBeCloseTo(expected[i]?.[1] ?? NaN, 6);
    });
  });

  it("follows the area centroid, not the mean of the corners", () => {
    // A square with an extra corner on its top edge: the corner mean moves up, the area
    // centroid does not.
    const [[x, y] = [NaN, NaN]] = centreZone(
      [[0, 0], [50, 0], [100, 0], [100, 100], [0, 100]], // prettier-ignore
      0.25,
    );
    expect([x, y]).toEqual([25, 25]);
  });

  it("is empty until the floor has three corners", () => {
    expect(centreZone(square.slice(0, 2), 0.5)).toEqual([]);
  });
});

describe("toImagePoint", () => {
  const box = { left: 100, top: 50, width: 720, height: 480 }; // the 360 x 240 frame at 2x

  it("maps a pointer on the scaled image to video pixels", () => {
    expect(toImagePoint(460, 290, box, 360, 240)).toEqual([180, 120]);
  });

  it("rounds to a tenth of a pixel: finer is noise from the scaling", () => {
    expect(toImagePoint(160.0000021, 110.33333, box, 360, 240)).toEqual([30, 30.2]);
  });

  it("keeps a corner dragged past the edge on the frame", () => {
    expect(toImagePoint(0, 1000, box, 360, 240)).toEqual([0, 240]);
  });
});

describe("moveCorner", () => {
  it("moves only the corner dragged", () => {
    expect(moveCorner(square, 2, [300, 300])).toEqual([
      [30, 30],
      [330, 30],
      [300, 300],
      [30, 330],
    ]);
    expect(square[2]).toEqual([330, 330]);
  });
});
