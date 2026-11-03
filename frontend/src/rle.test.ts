import { describe, expect, it } from "vitest";
import { decodeMask } from "./rle";

/** Pixels set in a decoded mask, as [row, col] pairs in row-major order. */
function setPixels(mask: Uint8Array, width: number): [number, number][] {
  const out: [number, number][] = [];
  mask.forEach((v, i) => {
    if (v) out.push([Math.floor(i / width), i % width]);
  });
  return out;
}

// Expected strings were encoded by pycocotools (the backend's encoder) from the masks described.
describe("decodeMask", () => {
  it("decodes a small COCO RLE into a row-major mask", () => {
    const mask = decodeMask("522O70", 4, 5);
    expect(setPixels(mask, 5)).toEqual([
      [1, 1],
      [1, 2],
      [2, 1],
      [3, 4],
    ]);
  });

  it("decodes long runs and repeated counts (delta-coded)", () => {
    // 60×80: rows 10–49, cols 5–69 set, plus pixel (0, 0).
    const rle =
      "01e9W1oF" + "0".repeat(127) + "^b0";
    const mask = decodeMask(rle, 60, 80);
    expect(mask.reduce((a, v) => a + v, 0)).toBe(2601);
    expect(mask[0]).toBe(1);
    expect(mask[10 * 80 + 5]).toBe(1);
    expect(mask[49 * 80 + 69]).toBe(1);
    expect(mask[50 * 80 + 69]).toBe(0);
    expect(mask[10 * 80 + 70]).toBe(0);
  });
});
