import type { Qc } from "./api";
import { frameSpan } from "./frames";
import { plural } from "./text";

/** Every exclusion in the run's QC artifact, so nothing is dropped silently. A window's frames
 * seek the video there. */
export function QcList({
  qc,
  fps,
  onSeek,
}: {
  qc: Qc;
  fps: number;
  onSeek: (frame: number) => void;
}) {
  const warnings = qc.flags.filter((f) => f.severity === "warning").length;
  const summary =
    `${plural(qc.flags.length, "QC flag")} (${String(qc.flags.length - warnings)} critical, ` +
    `${plural(warnings, "warning")}); ${String(qc.reviews.length)} reviewed by the model.`;
  return (
    <section className="qc">
      <h3>Quality control</h3>
      <p className="muted">{summary}</p>
      {qc.exclusions.length ? (
        <table aria-label="QC exclusions">
          <thead>
            <tr>
              <th scope="col">Excluded</th>
              <th scope="col">Frames</th>
              <th scope="col">Reason</th>
              <th scope="col">Decided by</th>
            </tr>
          </thead>
          <tbody>
            {qc.exclusions.map((x) => (
              <tr key={`${x.scope}-${String(x.first_frame)}-${String(x.last_frame)}`}>
                <td>{x.scope === "trial" ? "whole trial" : "window"}</td>
                <td className="number">
                  <button
                    type="button"
                    className="link"
                    onClick={() => {
                      onSeek(x.first_frame);
                    }}
                  >
                    {frameSpan(x.first_frame, x.last_frame, fps)}
                  </button>
                </td>
                <td>{x.reason}</td>
                <td>{x.source}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p>No exclusions: every frame enters the statistics.</p>
      )}
    </section>
  );
}
