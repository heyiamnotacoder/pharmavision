import { useEffect, useState } from "react";
import { api, getJson, type Report, type StudyView } from "./api";
import { ReportView } from "./ReportView";

async function loadStudy(studyId: string): Promise<[StudyView, Report | null]> {
  const view = await api.study(studyId);
  const url = view.artifacts.report;
  return [view, url === undefined ? null : await getJson<Report>(url)];
}

/** A study's status and, once its report job has run, the report. Mount with `key={studyId}`. */
export function StudyViewer({ studyId }: { studyId: string }) {
  const [study, setStudy] = useState<StudyView | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    loadStudy(studyId).then(
      ([view, shipped]) => {
        if (!live) return;
        setStudy(view);
        setReport(shipped);
      },
      (e: unknown) => {
        if (live) setError(String(e));
      },
    );
    return () => {
      live = false;
    };
  }, [studyId]);

  if (error)
    return (
      <p className="error">
        Could not show study {studyId}: {error}
      </p>
    );
  if (!study) return <p>Loading study {studyId}…</p>;
  return (
    <section className="viewer">
      <h2>Study {studyId}</h2>
      {report ? (
        <ReportView report={report} />
      ) : (
        <p className="missing">
          No report yet
          {study.status === null
            ? ": none has been asked for."
            : study.error
              ? `: ${study.status} at ${study.stages_done.at(-1) ?? "stats"}: ${study.error}`
              : ` (${study.status}).`}
        </p>
      )}
    </section>
  );
}
