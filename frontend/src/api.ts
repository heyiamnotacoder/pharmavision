/**
 * The run API (`src/pharmavision/api/app.py`) and the artifacts it serves. The types mirror the
 * Pydantic models in `artifacts.py`, keeping only the fields the frontend reads.
 */

/** A pixel position in the normalised video, as `measure.geometry.Point`. */
export type Point = [number, number];

export type JobStatus = "queued" | "running" | "awaiting zone confirmation" | "done" | "failed";

/** Statuses a run rests in until a person acts, or for good (as `jobs.RESTING`). */
export const RESTING: ReadonlySet<JobStatus> = new Set([
  "awaiting zone confirmation",
  "done",
  "failed",
]);

export interface RunView {
  run_id: string;
  status: JobStatus;
  /** The stage running now; after a failure, the stage that failed. */
  stage: string | null;
  error: string | null;
  read_only: boolean;
  stages_done: string[];
  /** The URL serving each artifact the run has written so far, by name. */
  artifacts: Partial<
    Record<
      "manifest" | "video" | "background" | "arena" | "track" | "metrics" | "events" | "qc",
      string
    >
  >;
}

export interface Manifest {
  /** The centre zone: the floor shrunk about its centroid to this fraction of its area. */
  template: { centre_area_fraction: number };
  measure: { version: number };
  video: { width: number; height: number; fps: number; n_frames: number };
}

/** Empty (no floor, no centre, `px_per_cm` 0) when SAM 3.1 found no floor to show. */
export interface Arena {
  floor_px: Point[];
  /** Derived from the floor by the backend; never drawn by hand. */
  centre_px: Point[];
  px_per_cm: number;
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

/** The arena gate: confirm the zones on `floor`, as the scientist left it. The API derives the
 * centre zone and the scale from it, then runs every later stage. A refusal throws with the
 * API's reason (a floor whose edges cross, a run no longer at the gate). */
async function confirmArena(runId: string, floor: readonly Point[]): Promise<RunView> {
  const path = `/runs/${encodeURIComponent(runId)}/arena/confirm`;
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ floor_px: floor }),
  });
  const body = (await response.json().catch(() => null)) as { detail?: unknown } | null;
  if (!response.ok) {
    const detail = body?.detail;
    throw new Error(
      typeof detail === "string" ? detail : `POST ${path}: ${String(response.status)}`,
    );
  }
  return body as RunView;
}

export const api = {
  runs: () => getJson<RunView[]>("/runs"),
  run: (id: string) => getJson<RunView>(`/runs/${encodeURIComponent(id)}`),
  confirmArena,
  studies: () => getJson<StudyView[]>("/studies"),
  study: (id: string) => getJson<StudyView>(`/studies/${encodeURIComponent(id)}`),
};
