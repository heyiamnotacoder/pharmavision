/** Tolerance, in frames, for a presentation time a hair under its frame's start. */
const EPSILON_FRAMES = 1e-3;

/**
 * The track record shown at `seconds` into the normalised video: overlays key on the
 * frame index, never on wall-clock time, so they stay on the frame the video is showing.
 */
export function frameAt(seconds: number, fps: number, nFrames: number): number {
  const index = Math.floor(seconds * fps + EPSILON_FRAMES);
  return Math.min(Math.max(index, 0), nFrames - 1);
}
