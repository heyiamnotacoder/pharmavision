import { useEffect, useState } from "react";
import { api, type RunView } from "./api";
import { RunViewer } from "./RunViewer";

/** `#/runs/<id>` opens a run; anything else lists them. No router needed for two screens. */
function runFromHash(hash: string): string | null {
  const match = /^#\/runs\/(.+)$/.exec(hash);
  return match?.[1] ? decodeURIComponent(match[1]) : null;
}

export function App() {
  const [runId, setRunId] = useState(() => runFromHash(window.location.hash));

  useEffect(() => {
    const onHash = () => {
      setRunId(runFromHash(window.location.hash));
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
      {runId ? <RunViewer key={runId} runId={runId} /> : <RunList />}
    </main>
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
