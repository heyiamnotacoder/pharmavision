import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { api, type Arena, type Point } from "./api";
import { COLOURS } from "./overlay";
import { centreZone, clampPoint, moveCorner, toImagePoint } from "./zones";

/** The corners a drawn floor gets: the open field is a rectangle. */
const DRAWN_CORNERS = 4;

/** Arrow keys move a focused corner this far, in video pixels, and Shift+arrow this far. */
const KEY_STEP_PX = 1;
const SHIFT_KEY_STEP_PX = 10;

const KEY_MOVES: Record<string, Point> = {
  ArrowLeft: [-1, 0],
  ArrowRight: [1, 0],
  ArrowUp: [0, -1],
  ArrowDown: [0, 1],
};

interface Props {
  runId: string;
  backgroundUrl: string;
  /** The normalised video's size: the arena's pixel coordinates are in it. */
  width: number;
  height: number;
  arena: Arena;
  /** The template's centre zone, as a fraction of the floor's area. */
  centreFraction: number;
  /** The API took the zones; the later stages are running. */
  onConfirmed: () => void;
}

const points = (polygon: readonly Point[]) =>
  polygon.map(([x, y]) => `${String(x)},${String(y)}`).join(" ");

/**
 * The arena gate: the detected floor over the background frame, its corners draggable (or, if
 * none was found, clicked in one by one), and the centre zone following it. Confirming sends
 * the floor only: the backend derives the centre zone and the scale from it.
 */
export function ZoneEditor({
  runId,
  backgroundUrl,
  width,
  height,
  arena,
  centreFraction,
  onConfirmed,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [floor, setFloor] = useState<Point[]>(arena.floor_px);
  const [drawing, setDrawing] = useState(arena.floor_px.length === 0);
  const [dragging, setDragging] = useState<number | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const centre = centreZone(floor, centreFraction);

  const pointerAt = (e: PointerEvent): Point | null => {
    const svg = svgRef.current;
    return svg
      ? toImagePoint(e.clientX, e.clientY, svg.getBoundingClientRect(), width, height)
      : null;
  };

  const onBackgroundDown = (e: PointerEvent<SVGSVGElement>) => {
    const at = pointerAt(e);
    if (!drawing || !at) return;
    const next = [...floor, at];
    setFloor(next);
    if (next.length >= DRAWN_CORNERS) setDrawing(false);
  };

  const onCornerDown = (e: PointerEvent<SVGCircleElement>, index: number) => {
    e.stopPropagation(); // not a click on the background, which would add a corner
    svgRef.current?.setPointerCapture(e.pointerId); // moves stay ours off the image
    setDragging(index);
  };

  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    const at = pointerAt(e);
    if (dragging !== null && at) setFloor((f) => moveCorner(f, dragging, at));
  };

  const onUp = () => {
    setDragging(null);
  };

  const onCornerKey = (e: KeyboardEvent<SVGCircleElement>, index: number) => {
    const move = KEY_MOVES[e.key];
    const corner = floor[index];
    if (!move || !corner) return;
    e.preventDefault();
    const step = e.shiftKey ? SHIFT_KEY_STEP_PX : KEY_STEP_PX;
    const to: Point = [corner[0] + move[0] * step, corner[1] + move[1] * step];
    setFloor((f) => moveCorner(f, index, clampPoint(to, width, height)));
  };

  const confirm = () => {
    setSending(true);
    setError(null);
    api.confirmArena(runId, floor).then(
      () => {
        onConfirmed();
      },
      (e: unknown) => {
        setError(e instanceof Error ? e.message : String(e));
        setSending(false);
      },
    );
  };

  const found = arena.floor_px.length > 0;
  return (
    <figure className="zone-editor">
      <p className="hint">
        {found
          ? "Check the floor against the arena's edge, and drag any corner that is off."
          : "SAM 3.1 found no floor on this video: click the arena floor's four corners."}{" "}
        The centre zone follows the floor; the scale (px/cm) is derived from it on confirming.
      </p>
      <div className="stage" style={{ aspectRatio: `${String(width)} / ${String(height)}` }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${String(width)} ${String(height)}`}
          aria-label="Zone editor"
          className={drawing ? "drawing" : undefined}
          onPointerDown={onBackgroundDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
        >
          <image href={backgroundUrl} width={width} height={height} />
          {floor.length >= 2 && (
            <polygon className="floor" points={points(floor)} stroke={COLOURS.floor} />
          )}
          {centre.length > 0 && (
            <polygon className="centre" points={points(centre)} stroke={COLOURS.centre} />
          )}
          {floor.map(([x, y], i) => (
            <circle
              key={i}
              className="corner"
              cx={x}
              cy={y}
              r={Math.max(width, height) / 90}
              role="button"
              tabIndex={0}
              aria-label={`floor corner ${String(i + 1)} at ${x.toFixed(0)}, ${y.toFixed(0)} px`}
              onPointerDown={(e) => {
                onCornerDown(e, i);
              }}
              onKeyDown={(e) => {
                onCornerKey(e, i);
              }}
            />
          ))}
        </svg>
      </div>
      <figcaption>
        <button type="button" onClick={confirm} disabled={sending || drawing || floor.length < 3}>
          Confirm zones
        </button>
        <button
          type="button"
          onClick={() => {
            setFloor([]);
            setDrawing(true);
          }}
          disabled={sending}
        >
          Redraw the floor
        </button>
        {found && (
          <button
            type="button"
            onClick={() => {
              setFloor(arena.floor_px);
              setDrawing(false);
            }}
            disabled={sending}
          >
            Back to the detected floor
          </button>
        )}
        {drawing && (
          <span className="missing">
            {" "}
            corner {floor.length + 1} of {DRAWN_CORNERS}
          </span>
        )}
        {error && (
          <p role="alert" className="error">
            Not confirmed: {error}
          </p>
        )}
      </figcaption>
    </figure>
  );
}
