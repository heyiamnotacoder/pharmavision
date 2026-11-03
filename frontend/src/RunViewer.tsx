import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Arena, type Manifest, type Metrics, type Track } from "./api";
import { frameAt } from "./frames";
import { COLOURS, drawPolygon, drawTrail, maskImage, trail, type Position } from "./overlay";
import { MetricsTable } from "./MetricsTable";

/** How much of the path the overlay keeps behind the animal. */
const TAIL_SECONDS = 10;

interface RunData {
  manifest: Manifest;
  arena: Arena;
  track: Track | null;
  metrics: Metrics | null;
}

/** An artifact a run has not written yet (its stage hasn't run) reads as null. */
async function optional<T>(name: string, artifacts: Record<string, string>, load: () => Promise<T>) {
  return name in artifacts ? load() : null;
}

async function loadRun(runId: string): Promise<RunData> {
  const view = await api.run(runId);
  const [manifest, arena, track, metrics] = await Promise.all([
    api.manifest(runId),
    api.arena(runId),
    optional("track", view.artifacts, () => api.track(runId)),
    optional("metrics", view.artifacts, () => api.metrics(runId)),
  ]);
  return { manifest, arena, track, metrics };
}

/** Mount with `key={runId}`: its state belongs to one run. */
export function RunViewer({ runId }: { runId: string }) {
  const [data, setData] = useState<RunData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    loadRun(runId).then(
      (d) => {
        if (live) setData(d);
      },
      (e: unknown) => {
        if (live) setError(String(e));
      },
    );
    return () => {
      live = false;
    };
  }, [runId]);

  if (error) return <p className="error">Could not load run {runId}: {error}</p>;
  if (!data) return <p>Loading run {runId}…</p>;
  return (
    <section className="viewer">
      <h2>Run {runId}</h2>
      <OverlayPlayer runId={runId} {...data} />
      {data.metrics ? (
        <MetricsTable metrics={data.metrics} measureVersion={data.manifest.measure.version} />
      ) : (
        <p className="missing">No metrics yet: the run has not reached the measure stage.</p>
      )}
    </section>
  );
}

function OverlayPlayer({
  runId,
  manifest,
  arena,
  track,
}: {
  runId: string;
  manifest: Manifest;
  arena: Arena;
  track: Track | null;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [frame, setFrame] = useState(0);
  const { width, height, fps, n_frames: nFrames } = manifest.video;

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
      drawPolygon(ctx, arena.floor_px, COLOURS.floor);
      drawPolygon(ctx, arena.centre_px, COLOURS.centre);
      drawTrail(ctx, trail(positions, index, tailFrames));
      setFrame(index);
    };
    let handle = 0;
    const onFrame = (_now: number, meta: VideoFrameCallbackMetadata) => {
      draw(meta.mediaTime);
      handle = video.requestVideoFrameCallback(onFrame);
    };
    const onSeek = () => {
      draw(video.currentTime);
    };
    draw(video.currentTime);
    handle = video.requestVideoFrameCallback(onFrame);
    video.addEventListener("seeked", onSeek);
    return () => {
      video.cancelVideoFrameCallback(handle);
      video.removeEventListener("seeked", onSeek);
    };
  }, [arena, track, positions, width, height, fps, nFrames]);

  const record = track?.frames[frame];
  return (
    <figure className="player">
      <div className="stage" style={{ aspectRatio: `${String(width)} / ${String(height)}` }}>
        <video ref={videoRef} src={api.videoUrl(runId)} controls muted playsInline />
        <canvas ref={canvasRef} width={width} height={height} aria-label="Tracking overlay" />
      </div>
      <figcaption>
        <span data-testid="frame">
          Frame {frame + 1} / {nFrames}
        </span>
        {track ? (
          record?.detected ? null : <span className="gap"> · animal not found (gap)</span>
        ) : (
          <span className="missing"> · not tracked yet</span>
        )}
        {arena.confirmed ? null : <span className="missing"> · zones not confirmed</span>}
        <span className="legend">
          <i style={{ borderColor: COLOURS.floor }} /> floor <i style={{ borderColor: COLOURS.centre }} />{" "}
          centre zone <i className="mask" /> mask <i className="tail" /> last {TAIL_SECONDS} s
        </span>
      </figcaption>
    </figure>
  );
}
