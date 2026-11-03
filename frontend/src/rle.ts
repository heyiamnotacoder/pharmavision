/**
 * COCO RLE masks as the track stage writes them (`stages/track.py`): pycocotools' compressed
 * counts string, column-major, size from the manifest's normalised video.
 */

/** The run lengths in a compressed counts string (pycocotools `rleFrString`). */
function runLengths(counts: string): number[] {
  const runs: number[] = [];
  let p = 0;
  while (p < counts.length) {
    let x = 0;
    let k = 0;
    let more = true;
    while (more) {
      const c = counts.charCodeAt(p) - 48;
      x |= (c & 0x1f) << (5 * k);
      more = (c & 0x20) !== 0;
      p++;
      k++;
      if (!more && (c & 0x10) !== 0) x |= -1 << (5 * k);
    }
    const before = runs[runs.length - 2];
    runs.push(runs.length > 2 && before !== undefined ? x + before : x);
  }
  return runs;
}

/** The mask as 0/1 per pixel, row-major (index `row * width + col`), for drawing. */
export function decodeMask(counts: string, height: number, width: number): Uint8Array {
  const mask = new Uint8Array(height * width);
  let i = 0;
  let value = 0;
  for (const run of runLengths(counts)) {
    if (value) {
      for (let j = i; j < i + run; j++) {
        const col = Math.floor(j / height);
        const row = j % height;
        mask[row * width + col] = 1;
      }
    }
    i += run;
    value ^= 1;
  }
  return mask;
}
