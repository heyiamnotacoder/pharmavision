/**
 * What the run viewer draws over the video: the arena zones, the animal's mask and a fading
 * tail of its path. The canvas has the normalised video's pixel size, so artifact pixel
 * coordinates are canvas coordinates; CSS scales both together.
 */
import { decodeMask } from "./rle";

export interface Position {
  x: number;
  y: number;
}

export interface TrailPoint extends Position {
  /** 1 for the oldest point kept, 0 for the current frame. */
  age: number;
}

/**
 * The path over the last `tailFrames` frames up to `current`, as polylines: a gap (null)
 * ends one and the next detection starts another, so missing data is never drawn as motion.
 */
export function trail(
  positions: readonly (Position | null)[],
  current: number,
  tailFrames: number,
): TrailPoint[][] {
  const lines: TrailPoint[][] = [];
  let line: TrailPoint[] = [];
  const span = Math.max(tailFrames - 1, 1);
  for (let i = Math.max(current - tailFrames + 1, 0); i <= current; i++) {
    const pos = positions[i];
    if (pos) {
      line.push({ x: pos.x, y: pos.y, age: (current - i) / span });
    } else if (line.length) {
      lines.push(line);
      line = [];
    }
  }
  if (line.length) lines.push(line);
  return lines;
}

export const COLOURS = {
  floor: "rgba(80, 160, 255, 0.9)",
  centre: "rgba(255, 200, 60, 0.9)",
  mask: [255, 60, 120, 110] as const,
  trail: "255, 255, 255",
};

export function drawPolygon(
  ctx: CanvasRenderingContext2D,
  polygon: readonly (readonly [number, number])[],
  colour: string,
): void {
  if (!polygon.length) return;
  ctx.beginPath();
  polygon.forEach(([x, y], i) => {
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.strokeStyle = colour;
  ctx.lineWidth = 3;
  ctx.stroke();
}

export function drawTrail(ctx: CanvasRenderingContext2D, lines: TrailPoint[][]): void {
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  for (const line of lines) {
    for (let i = 1; i < line.length; i++) {
      const a = line[i - 1];
      const b = line[i];
      if (!a || !b) continue;
      ctx.strokeStyle = `rgba(${COLOURS.trail}, ${(1 - b.age).toFixed(3)})`;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  }
}

/** The mask as an RGBA image the canvas size of the video, for `putImageData`. */
export function maskImage(counts: string, width: number, height: number): ImageData {
  const mask = decodeMask(counts, height, width);
  const image = new ImageData(width, height);
  const [r, g, b, a] = COLOURS.mask;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    const o = i * 4;
    image.data[o] = r;
    image.data[o + 1] = g;
    image.data[o + 2] = b;
    image.data[o + 3] = a;
  }
  return image;
}
