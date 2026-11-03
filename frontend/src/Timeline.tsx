import type { ReactNode } from "react";
import type { BehaviourEvent, Exclusion } from "./api";
import { frameSpan, windowKey } from "./frames";

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

/** One window on a lane: its frames (inclusive), its look, and what it says. */
interface Mark {
  first_frame: number;
  last_frame: number;
  className: string;
  label: string;
  title: string;
}

/** The trial's frames as a bar per lane: each behaviour event, and each exclusion. A click
 * seeks the video to the window's first frame. */
export function Timeline({ events, exclusions, nFrames, fps, frame, onSeek }: TimelineProps) {
  const lane = (name: string, marks: Mark[] | null, missing: string) => (
    <Lane name={name} missing={missing} nFrames={nFrames} frame={frame} onSeek={onSeek}>
      {marks}
    </Lane>
  );
  return (
    <section className="timeline" aria-label="Event timeline">
      {lane(
        "Behaviour events",
        events?.map((event) => {
          const confidence = `confidence ${event.confidence.toFixed(2)}`;
          return {
            ...event,
            className: `event ${event.label}${event.low_confidence ? " low" : ""}`,
            label: `${event.label}, ${frameSpan(event.first_frame, event.last_frame, fps)}, ${confidence}${event.low_confidence ? " (low)" : ""}`,
            title: `${event.label} (${confidence}, ${event.source}): ${event.rationale}`,
          };
        }) ?? null,
        "not judged yet: the run has not reached the behaviours stage",
      )}
      {lane(
        "Exclusions",
        exclusions?.map((exclusion) => ({
          ...exclusion,
          className: `exclusion ${exclusion.scope}`,
          label: `excluded: ${exclusion.scope === "trial" ? "whole trial" : "window"}, ${frameSpan(exclusion.first_frame, exclusion.last_frame, fps)}`,
          title: `${exclusion.source}: ${exclusion.reason}`,
        })) ?? null,
        "no QC yet: the run has not reached the QC stage",
      )}
    </section>
  );
}

/** Frames `first` to `last` (inclusive) as a stretch of the bar, in percent of its width. */
function spanStyle(first: number, last: number, nFrames: number) {
  return {
    left: `${String((first / nFrames) * 100)}%`,
    width: `${String(((last - first + 1) / nFrames) * 100)}%`,
  };
}

function Lane({
  name,
  missing,
  nFrames,
  frame,
  onSeek,
  children: marks,
}: {
  name: string;
  /** Said instead of the bar when the lane's artifact is not written yet. */
  missing: string;
  nFrames: number;
  frame: number;
  onSeek: (frame: number) => void;
  children: Mark[] | null;
}): ReactNode {
  return (
    <div className="lane">
      <span className="lane-name">{name}</span>
      {marks ? (
        <div className="bar">
          {marks.map((mark) => (
            <button
              key={windowKey(mark)}
              type="button"
              className={mark.className}
              style={spanStyle(mark.first_frame, mark.last_frame, nFrames)}
              aria-label={mark.label}
              title={mark.title}
              onClick={() => {
                onSeek(mark.first_frame);
              }}
            />
          ))}
          <i className="playhead" style={{ left: `${String(((frame + 0.5) / nFrames) * 100)}%` }} />
        </div>
      ) : (
        <span className="missing">{missing}</span>
      )}
    </div>
  );
}
