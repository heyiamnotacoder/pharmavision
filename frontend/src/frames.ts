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

/**
 * Where to seek the video to show frame `frame`: the middle of its display interval. Seeking to
 * the frame's exact start can land a hair before it and show the frame before.
 */
export function seekTime(frame: number, fps: number): number {
  return (frame + 0.5) / fps;
}

/** A time into the video as the player's controls show it, to a tenth: `3:20.0`. */
export function clock(seconds: number): string {
  const tenths = Math.round(seconds * 10);
  const minutes = Math.floor(tenths / 600);
  const rest = ((tenths % 600) / 10).toFixed(1).padStart(4, "0");
  return `${String(minutes)}:${rest}`;
}

/** A key for a window of frames, unique among a list's windows. */
export function windowKey(w: { first_frame: number; last_frame: number; scope?: string }): string {
  return `${w.scope ?? ""}${String(w.first_frame)}-${String(w.last_frame)}`;
}

/** Frames `first` to `last` (inclusive), and the stretch of video they span. */
export function frameSpan(first: number, last: number, fps: number): string {
  return `frames ${String(first)}–${String(last)} (${clock(first / fps)}–${clock((last + 1) / fps)})`;
}
