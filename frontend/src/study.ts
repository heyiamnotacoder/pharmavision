import type { StudyView } from "./api";

/** Where a study's report stands, in a few words, for the study list and the study screen. */
export function reportStatus(study: StudyView): string {
  if (study.artifacts.report !== undefined) return "ready";
  if (study.status === null) return "not asked for yet";
  if (study.status === "failed")
    return `failed at ${study.stage ?? "an unknown stage"}: ${study.error ?? "no reason given"}`;
  return study.stage ? `${study.status}: ${study.stage}` : study.status;
}
