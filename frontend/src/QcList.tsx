import type { Qc, QcFlag } from "./api";
import { frameSpan, windowKey } from "./frames";
import { plural } from "./plural";

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
  const count = (severity: QcFlag["severity"]) =>
    qc.flags.filter((flag) => flag.severity === severity).length;
  const warnings = count("warning");
  const summary =
    `${plural(qc.flags.length, "QC flag")} (${String(count("critical"))} critical, ` +
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
            {qc.exclusions.map((exclusion) => (
              <tr key={windowKey(exclusion)}>
                <td>{exclusion.scope === "trial" ? "whole trial" : "window"}</td>
                <td className="number">
                  <button
                    type="button"
                    className="link"
                    onClick={() => {
                      onSeek(exclusion.first_frame);
                    }}
                  >
                    {frameSpan(exclusion.first_frame, exclusion.last_frame, fps)}
                  </button>
                </td>
                <td>{exclusion.reason}</td>
                <td>{exclusion.source}</td>
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
