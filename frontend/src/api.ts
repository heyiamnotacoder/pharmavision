/**
 * The run API (`src/pharmavision/api/app.py`) and the artifacts it serves. The types mirror the
 * Pydantic models in `artifacts.py`, keeping only the fields the frontend reads.
 */

/** A pixel position in the normalised video, as `measure.geometry.Point`. */
export type Point = [number, number];

export type JobStatus = "queued" | "running" | "awaiting zone confirmation" | "done" | "failed";

export interface RunView {
  run_id: string;
  status: JobStatus;
  error: string | null;
  read_only: boolean;
  stages_done: string[];
  /** The URL serving each artifact the run has written so far, by name. */
  artifacts: Partial<Record<"manifest" | "video" | "arena" | "track" | "metrics", string>>;
}

export interface Manifest {
  measure: { version: number };
  video: { width: number; height: number; fps: number; n_frames: number };
}

export interface Arena {
  floor_px: Point[];
  centre_px: Point[];
  confirmed: boolean;
}

export interface TrackFrame {
  detected: boolean;
  x_px: number | null;
  y_px: number | null;
  mask_rle: string | null;
}

export interface Track {
  /** One record per frame: the backend's `Track` validator holds `frame_index` to the
   * record's position, so `frames[i]` is frame i. */
  frames: TrackFrame[];
}

export interface Metrics {
  distance_travelled_cm: number | null;
  mean_velocity_cm_s: number | null;
  centre_time_s: number | null;
  centre_entries: number | null;
  thigmotaxis: number | null;
  immobility_s: number | null;
}

/** The API's root: `/api` goes through the Vite dev proxy; a deploy can point elsewhere. */
const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

/** A path the API gave (such as a `RunView.artifacts` URL), as this page reaches it. */
export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path));
  if (!response.ok) throw new Error(`GET ${path}: ${String(response.status)}`);
  return (await response.json()) as T;
}

export const api = {
  runs: () => getJson<RunView[]>("/runs"),
  run: (id: string) => getJson<RunView>(`/runs/${encodeURIComponent(id)}`),
};
