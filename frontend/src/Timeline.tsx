import type { BehaviourEvent, Exclusion } from "./api";
import { frameSpan } from "./frames";

interface TimelineProps {
  /** Null until the behaviours stage has run; likewise `exclusions` and the QC stage. */
  events: BehaviourEvent[] | null;
  exclusions: Exclusion[] | null;
  nFrames: number;
  fps: number;
  /** The frame the video shows, for the playhead. */
  frame: number;
  onSeek: (frame: number) => void;
}

/** Frames `first` to `last` (inclusive) as a stretch of the bar, in percent of its width. */
function place(first: number, last: number, nFrames: number) {
  return {
    left: `${String((first / nFrames) * 100)}%`,
    width: `${String(((last - first + 1) / nFrames) * 100)}%`,
  };
}

/** The trial's frames as a bar per lane: each judged rear candidate, and each exclusion. A
 * click seeks the video to the window's first frame. */
export function Timeline({ events, exclusions, nFrames, fps, frame, onSeek }: TimelineProps) {
  const playhead = (
    <i className="playhead" style={{ left: `${String(((frame + 0.5) / nFrames) * 100)}%` }} />
  );
  return (
    <section className="timeline" aria-label="Event timeline">
      <div className="lane">
        <span className="lane-name">Rear candidates</span>
        {events ? (
          <div className="bar">
            {events.map((e) => {
              const span = frameSpan(e.first_frame, e.last_frame, fps);
              const confidence = `confidence ${e.confidence.toFixed(2)}`;
              return (
                <button
                  key={e.first_frame}
                  type="button"
                  className={`event ${e.label}${e.low_confidence ? " low" : ""}`}
                  style={place(e.first_frame, e.last_frame, nFrames)}
                  aria-label={`${e.label}, ${span}, ${confidence}${e.low_confidence ? " (low)" : ""}`}
                  title={`${e.label} (${confidence}, ${e.source}): ${e.rationale}`}
                  onClick={() => {
                    onSeek(e.first_frame);
                  }}
                />
              );
            })}
            {playhead}
          </div>
        ) : (
          <span className="missing">
            not judged yet: the run has not reached the behaviours stage
          </span>
        )}
      </div>
      <div className="lane">
        <span className="lane-name">Exclusions</span>
        {exclusions ? (
          <div className="bar">
            {exclusions.map((x) => (
              <button
                key={`${x.scope}-${String(x.first_frame)}`}
                type="button"
                className={`exclusion ${x.scope}`}
                style={place(x.first_frame, x.last_frame, nFrames)}
                aria-label={`excluded, ${frameSpan(x.first_frame, x.last_frame, fps)}`}
                title={`${x.source}: ${x.reason}`}
                onClick={() => {
                  onSeek(x.first_frame);
                }}
              />
            ))}
            {playhead}
          </div>
        ) : (
          <span className="missing">no QC yet: the run has not reached the QC stage</span>
        )}
      </div>
    </section>
  );
}
