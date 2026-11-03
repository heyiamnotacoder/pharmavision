import { useEffect, useState } from "react";
import { api, type RunView, type StudyView } from "./api";
import { routeFromHash } from "./route";
import { RunViewer } from "./RunViewer";
import { StudyViewer } from "./StudyViewer";

export function App() {
  const [route, setRoute] = useState(() => routeFromHash(window.location.hash));

  useEffect(() => {
    const onHash = () => {
      setRoute(routeFromHash(window.location.hash));
    };
    window.addEventListener("hashchange", onHash);
    return () => {
      window.removeEventListener("hashchange", onHash);
    };
  }, []);

  return (
    <main>
      <header>
        <h1>
          <a href="#/">PharmaVision</a>
        </h1>
        <p>Rodent test video in, scored and audited results out.</p>
      </header>
      {route.screen === "run" ? (
        <RunViewer key={route.id} runId={route.id} />
      ) : route.screen === "study" ? (
        <StudyViewer key={route.id} studyId={route.id} />
      ) : (
        <>
          <StudyList />
          <RunList />
        </>
      )}
    </main>
  );
}

function StudyList() {
  const [studies, setStudies] = useState<StudyView[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.studies().then(setStudies, (e: unknown) => {
      setError(String(e));
    });
  }, []);

  if (error) return <p className="error">Could not list the studies: {error}</p>;
  if (!studies) return <p>Loading studies…</p>;
  if (!studies.length) return null;
  return (
    <table className="runs" aria-label="Studies">
      <thead>
        <tr>
          <th scope="col">Study</th>
          <th scope="col">Report</th>
        </tr>
      </thead>
      <tbody>
        {studies.map((study) => (
          <tr key={study.study_id}>
            <td>
              <a href={`#/studies/${encodeURIComponent(study.study_id)}`}>{study.study_id}</a>
            </td>
            <td>
              {study.artifacts.report
                ? "ready"
                : study.status === null
                  ? "not asked for"
                  : study.error
                    ? `${study.status}: ${study.error}`
                    : study.status}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function RunList() {
  const [runs, setRuns] = useState<RunView[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.runs().then(setRuns, (e: unknown) => {
      setError(String(e));
    });
  }, []);

  if (error) return <p className="error">Could not reach the API: {error}</p>;
  if (!runs) return <p>Loading runs…</p>;
  if (!runs.length) return <p>No runs yet.</p>;
  return (
    <table className="runs" aria-label="Runs">
      <thead>
        <tr>
          <th scope="col">Run</th>
          <th scope="col">Status</th>
          <th scope="col">Stages done</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((run) => (
          <tr key={run.run_id}>
            <td>
              <a href={`#/runs/${encodeURIComponent(run.run_id)}`}>{run.run_id}</a>
              {run.read_only ? <span className="tag">precomputed</span> : null}
            </td>
            <td>{run.error ? `${run.status}: ${run.error}` : run.status}</td>
            <td>{run.stages_done.join(" → ")}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
