import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import {
  api,
  apiUrl,
  RESTING,
  getJson,
  type Arena,
  type Events,
  type Manifest,
  type Metrics,
  type Qc,
  type RunView,
  type Track,
} from "./api";
import { clock, frameAt, seekTime } from "./frames";
import { COLOURS, drawPolygon, drawPathTail, maskImage, pathTail, type Position } from "./overlay";
import { MetricsTable } from "./MetricsTable";
import { QcList } from "./QcList";
import { Timeline } from "./Timeline";
import { ZoneEditor } from "./ZoneEditor";

/** How much of the path the overlay keeps behind the animal. */
const TAIL_SECONDS = 10;

/** A run that has been ingested: its manifest and video exist. Each later artifact is null
 * until its stage has run, and the viewer says so instead of failing. */
interface RunData {
  view: RunView;
  manifest: Manifest;
  videoUrl: string;
  arena: Arena | null;
  track: Track | null;
  metrics: Metrics | null;
  events: Events | null;
  qc: Qc | null;
}

class NotIngestedError extends Error {}

/** The artifact at the URL the run view gives, or null if the run has not written it. */
async function optional<T>(url: string | undefined): Promise<T | null> {
  return url === undefined ? null : getJson<T>(url);
}

async function loadRun(runId: string): Promise<RunData> {
  const view = await api.run(runId);
  const { artifacts } = view;
  if (artifacts.manifest === undefined || artifacts.video === undefined) {
    throw new NotIngestedError(`run ${runId} has not been ingested yet`);
  }
  const [manifest, arena, track, metrics, events, qc] = await Promise.all([
    getJson<Manifest>(artifacts.manifest),
    optional<Arena>(artifacts.arena),
    optional<Track>(artifacts.track),
    optional<Metrics>(artifacts.metrics),
    optional<Events>(artifacts.events),
    optional<Qc>(artifacts.qc),
  ]);
  return { view, manifest, videoUrl: apiUrl(artifacts.video), arena, track, metrics, events, qc };
}

/** Mount with `key={runId}`: its state belongs to one run. */
export function RunViewer({ runId }: { runId: string }) {
  const [data, setData] = useState<RunData | null>(null);
  const [error, setError] = useState<string | null>(null);
  /** Bumped to load the run again: once its zones are confirmed, and as its stages finish. */
  const [loads, setLoads] = useState(0);
  const reload = useCallback(() => {
    setLoads((n) => n + 1);
  }, []);
  /** This page confirmed the zones: the job writes the confirmed arena a moment later. */
  const [confirmedHere, setConfirmedHere] = useState(false);
  const onConfirmed = useCallback(() => {
    setConfirmedHere(true);
    reload();
  }, [reload]);
  const running = data !== null && !RESTING.has(data.view.status);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fps = data?.manifest.video.fps;
  const seek = useCallback(
    (frame: number) => {
      if (videoRef.current && fps) videoRef.current.currentTime = seekTime(frame, fps);
    },
    [fps],
  );

  useEffect(() => {
    let live = true;
    loadRun(runId).then(
      (d) => {
        if (live) setData(d);
      },
      (e: unknown) => {
        if (live) setError(e instanceof NotIngestedError ? e.message : String(e));
      },
    );
    return () => {
      live = false;
    };
  }, [runId, loads]);

  // While the run's stages run, each one it finishes (and the status it rests in) loads it
  // again, so the viewer fills in as artifacts appear.
  useEffect(() => {
    if (!running || typeof EventSource === "undefined") return;
    const source = new EventSource(apiUrl(`/runs/${encodeURIComponent(runId)}/stream`));
    source.addEventListener("stage", reload);
    source.addEventListener("status", () => {
      source.close();
      reload();
    });
    return () => {
      source.close();
    };
  }, [runId, running, reload]);

  if (error)
    return (
      <p className="error">
        Could not show run {runId}: {error}
      </p>
    );
  if (!data) return <p>Loading run {runId}…</p>;
  const { view, arena, manifest } = data;
  const backgroundUrl = view.artifacts.background;
  const atGate = view.status === "awaiting zone confirmation" && !view.read_only;
  return (
    <section className="viewer">
      <h2>Run {runId}</h2>
      {atGate && arena && backgroundUrl ? (
        <ZoneEditor
          runId={runId}
          backgroundUrl={apiUrl(backgroundUrl)}
          width={manifest.video.width}
          height={manifest.video.height}
          arena={arena}
          centreFraction={manifest.template.centre_area_fraction}
          onConfirmed={onConfirmed}
        />
      ) : null}
      {running && (confirmedHere || arena?.confirmed) ? (
        <p className="status">
          Zones confirmed: {view.stage ?? "the next stage"} is running ({view.stages_done.length}{" "}
          stages done).
        </p>
      ) : null}
      {view.status === "failed" ? (
        <p className="error">
          The run failed{view.stage ? ` at ${view.stage}` : ""}: {view.error}
        </p>
      ) : null}
      <OverlayPlayer {...data} videoRef={videoRef} onSeek={seek} />
      {data.metrics ? (
        <MetricsTable metrics={data.metrics} measureVersion={data.manifest.measure.version} />
      ) : (
        <p className="missing">No metrics yet: the run has not reached the measure stage.</p>
      )}
      {data.qc ? (
        <QcList qc={data.qc} fps={data.manifest.video.fps} onSeek={seek} />
      ) : (
        <p className="missing">No QC yet: the run has not reached the QC stage.</p>
      )}
    </section>
  );
}

interface PlayerProps extends Omit<RunData, "metrics"> {
  videoRef: RefObject<HTMLVideoElement | null>;
  onSeek: (frame: number) => void;
}

function OverlayPlayer({
  manifest,
  videoUrl,
  arena,
  track,
  events,
  qc,
  videoRef,
  onSeek,
}: PlayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [frame, setFrame] = useState(0);
  const { width, height, fps, n_frames: nFrames } = manifest.video;

  // Indexed by frame: `Track.frames[i]` is frame i (see `Track` in api.ts).
  const positions = useMemo<(Position | null)[]>(
    () =>
      track?.frames.map((f) =>
        f.detected && f.x_px !== null && f.y_px !== null ? { x: f.x_px, y: f.y_px } : null,
      ) ?? [],
    [track],
  );

  // Draw each frame's overlay inside the video's frame callback, which runs before that frame
  // is composited: going through React state would land the overlay a frame late. The
  // caption's frame number goes through state; it can afford to.
  useEffect(() => {
    const video = videoRef.current;
    const ctx = canvasRef.current?.getContext("2d");
    if (!video || !ctx) return;
    const tailFrames = Math.round(TAIL_SECONDS * fps);
    let drawn = -1;
    const draw = (seconds: number) => {
      const index = frameAt(seconds, fps, nFrames);
      if (index === drawn) return;
      drawn = index;
      ctx.clearRect(0, 0, width, height);
      const record = track?.frames[index];
      if (record?.mask_rle) ctx.putImageData(maskImage(record.mask_rle, width, height), 0, 0);
      if (arena) {
        drawPolygon(ctx, arena.floor_px, COLOURS.floor);
        drawPolygon(ctx, arena.centre_px, COLOURS.centre);
      }
      drawPathTail(ctx, pathTail(positions, index, tailFrames));
      setFrame(index);
    };
    let handle = 0;
    const onFrame = (_now: number, meta: VideoFrameCallbackMetadata) => {
      draw(meta.mediaTime);
      handle = video.requestVideoFrameCallback(onFrame);
    };
    const onSeeked = () => {
      draw(video.currentTime);
    };
    draw(video.currentTime);
    handle = video.requestVideoFrameCallback(onFrame);
    video.addEventListener("seeked", onSeeked);
    return () => {
      video.cancelVideoFrameCallback(handle);
      video.removeEventListener("seeked", onSeeked);
    };
  }, [videoRef, arena, track, positions, width, height, fps, nFrames]);

  const record = track?.frames[frame];
  return (
    <figure className="player">
      <div className="stage" style={{ aspectRatio: `${String(width)} / ${String(height)}` }}>
        <video ref={videoRef} src={videoUrl} controls muted playsInline />
        <canvas ref={canvasRef} width={width} height={height} aria-label="Tracking overlay" />
      </div>
      <figcaption>
        <span data-testid="frame">
          Frame {frame} of 0–{nFrames - 1} · {clock(frame / fps)}
        </span>
        {track ? (
          record?.detected ? null : (
            <span className="gap"> · animal not found (gap)</span>
          )
        ) : (
          <span className="missing"> · not tracked yet</span>
        )}
        {!arena ? (
          <span className="missing"> · no zones yet</span>
        ) : arena.confirmed ? null : (
          <span className="missing"> · zones not confirmed</span>
        )}
        <span className="legend">
          <span>
            <i style={{ borderColor: COLOURS.floor }} />
            floor
          </span>
          <span>
            <i style={{ borderColor: COLOURS.centre }} />
            centre zone
          </span>
          <span>
            <i className="mask" />
            mask
          </span>
          <span title="Raw tracked centroids: the metrics use a smoothed path">
            <i className="tail" />
            path, last {TAIL_SECONDS} s (raw)
          </span>
        </span>
      </figcaption>
      <Timeline
        events={events?.events ?? null}
        exclusions={qc?.exclusions ?? null}
        nFrames={nFrames}
        fps={fps}
        frame={frame}
        onSeek={onSeek}
      />
    </figure>
  );
}
