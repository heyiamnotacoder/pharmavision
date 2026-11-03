/**
 * The run API (`src/pharmavision/api/app.py`) and the artifacts it serves. The types mirror the
 * Pydantic models in `artifacts.py`, keeping only the fields the frontend reads.
 */

export type Point = [number, number];

export type JobStatus = "queued" | "running" | "awaiting zone confirmation" | "done" | "failed";

export interface RunView {
  run_id: string;
  status: JobStatus;
  stage: string | null;
  error: string | null;
  read_only: boolean;
  stages_done: string[];
  artifacts: Record<string, string>;
}

export interface Manifest {
  run_id: string;
  code_version: string;
  measure: { version: number };
  video: { width: number; height: number; fps: number; duration_s: number; n_frames: number };
}

export interface Arena {
  floor_px: Point[];
  centre_px: Point[];
  px_per_cm: number;
  confirmed: boolean;
}

export interface TrackFrame {
  frame_index: number;
  detected: boolean;
  x_px: number | null;
  y_px: number | null;
  mask_rle: string | null;
}

export interface Track {
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

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path));
  if (!response.ok) throw new Error(`GET ${path}: ${String(response.status)}`);
  return (await response.json()) as T;
}

export const api = {
  runs: () => getJson<RunView[]>("/runs"),
  run: (id: string) => getJson<RunView>(`/runs/${encodeURIComponent(id)}`),
  manifest: (id: string) => getJson<Manifest>(`/runs/${encodeURIComponent(id)}/artifacts/manifest`),
  arena: (id: string) => getJson<Arena>(`/runs/${encodeURIComponent(id)}/artifacts/arena`),
  track: (id: string) => getJson<Track>(`/runs/${encodeURIComponent(id)}/artifacts/track`),
  metrics: (id: string) => getJson<Metrics>(`/runs/${encodeURIComponent(id)}/artifacts/metrics`),
  videoUrl: (id: string) => apiUrl(`/runs/${encodeURIComponent(id)}/video`),
};
