/**
 * The zone editor's geometry, in pixels of the normalised video. The floor is the only zone a
 * person moves: the centre zone shown beside it is a preview of what the backend derives
 * (`measure.geometry.shrink_to_area_fraction`) when the floor is confirmed, never sent back.
 */
import type { Point } from "./api";

/** The floor shrunk about its area centroid to `fraction` of its area. Empty for fewer than
 * three corners or no area: there is no centre to show yet. */
export function centreZone(floor: readonly Point[], fraction: number): Point[] {
  if (floor.length < 3) return [];
  let area = 0;
  let cx = 0;
  let cy = 0;
  floor.forEach(([x0, y0], i) => {
    const [x1, y1] = floor[(i + 1) % floor.length] ?? [x0, y0];
    const cross = x0 * y1 - x1 * y0;
    area += cross / 2;
    cx += (x0 + x1) * cross;
    cy += (y0 + y1) * cross;
  });
  if (area === 0) return [];
  cx /= 6 * area;
  cy /= 6 * area;
  const k = Math.sqrt(fraction);
  return floor.map(([x, y]) => [cx + k * (x - cx), cy + k * (y - cy)]);
}

/** The part of a bounding box `toImagePoint` reads (a `DOMRect` has it). */
export interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** A pointer position on the image, drawn scaled into `box`, in video pixels to a tenth of a
 * pixel, kept on the `width` x `height` frame. */
export function toImagePoint(
  clientX: number,
  clientY: number,
  box: Box,
  width: number,
  height: number,
): Point {
  const clamp = (v: number, max: number) => Math.min(Math.max(Math.round(v * 10) / 10, 0), max);
  return [
    clamp(((clientX - box.left) / box.width) * width, width),
    clamp(((clientY - box.top) / box.height) * height, height),
  ];
}

/** The floor with corner `index` at `to`; the floor given is left as it was. */
export function moveCorner(floor: readonly Point[], index: number, to: Point): Point[] {
  return floor.map((p, i) => (i === index ? to : p));
}
