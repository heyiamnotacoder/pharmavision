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
  artifacts: Partial<
    Record<"manifest" | "video" | "arena" | "track" | "metrics" | "events" | "qc", string>
  >;
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

/** One judged rear candidate over frames `first_frame` to `last_frame` (inclusive). */
export interface BehaviourEvent {
  first_frame: number;
  last_frame: number;
  label: "rearing" | "none";
  confidence: number;
  low_confidence: boolean;
  rationale: string;
  source: "rule" | "model";
}

export interface Events {
  /** In frame order. */
  events: BehaviourEvent[];
}

/** Frames (inclusive), or the whole trial, left out of the statistics, and why. */
export interface Exclusion {
  scope: "trial" | "window";
  first_frame: number;
  last_frame: number;
  reason: string;
  source: "rule" | "model";
}

export interface QcFlag {
  kind: "tracking_loss" | "out_of_arena" | "speed_jump" | "second_mask";
  first_frame: number;
  last_frame: number;
  severity: "warning" | "critical";
  reason: string;
}

/** The models' look at one flagged window, and the verdict (as `artifacts.QcReview`). */
export interface QcReview {
  flag: QcFlag;
  verdict: "keep" | "exclude_window" | "exclude_trial";
  reason: string;
}

export interface Qc {
  flags: QcFlag[];
  exclusions: Exclusion[];
  /** One per warning the models reviewed, in flag order. */
  reviews: QcReview[];
}

/** One number in the report prose, and the computed value it matched: `source` null when none
 * did, and its sentence is struck out of the report. */
export interface AuditedNumber {
  text: string;
  sentence: string;
  /** The sentence's start and end offsets in the draft's report. */
  sentence_span: [number, number];
  source: string | null;
}

export interface NumberAudit {
  /** In the order they appear. */
  numbers: AuditedNumber[];
}

export interface ReportDraft {
  audit: NumberAudit;
}

export interface Report {
  model_id: string;
  /** Every draft the number audit saw; the last one is the report. */
  drafts: ReportDraft[];
  /** The report as shipped: `[TO FILL: …]` placeholders for what the system cannot know, and
   * each sentence that failed the audit struck out and flagged. */
  markdown: string;
}

export interface StudyView {
  study_id: string;
  /** Null: no report has been asked for yet. */
  status: JobStatus | null;
  /** The stage running now; after a failure, the stage that failed. */
  stage: string | null;
  error: string | null;
  stages_done: string[];
  artifacts: Partial<Record<"design" | "trials" | "stats" | "report", string>>;
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
  studies: () => getJson<StudyView[]>("/studies"),
  study: (id: string) => getJson<StudyView>(`/studies/${encodeURIComponent(id)}`),
};
