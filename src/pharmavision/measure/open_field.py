"""Open field metrics: the track and the confirmed arena to one trial's numbers."""

from ..artifacts import Arena, MeasureSettings, Metrics, NormalisedVideo, Track
from .distance import distance_travelled_cm, mean_velocity_cm_s, speeds_cm_s
from .geometry import Point, signed_distance
from .immobility import immobility_s
from .rears import rear_candidates
from .shape import mask_changes, mask_elongations
from .trajectory import clean_trajectory
from .zones import zone_occupancy


def open_field_metrics(
    track: Track, arena: Arena, video: NormalisedVideo, settings: MeasureSettings
) -> Metrics:
    """Every open field metric for one trial. Zone and wall metrics cover the frames where the
    position is known after gap interpolation; thigmotaxis is the fraction of them within
    `thigmotaxis_wall_cm` of the floor edge (or beyond it)."""
    fps = video.fps
    trajectory = clean_trajectory(
        [f.position for f in track.frames],
        fps,
        settings.max_gap_s,
        settings.median_window_frames,
    )
    known = [p for p in trajectory.points if p is not None]
    centre = zone_occupancy(
        _per_frame(trajectory.points, signed_distance(known, arena.centre_px)),
        fps,
        settings.zone_hysteresis_cm * arena.px_per_cm,
        settings.min_dwell_s,
    )
    wall_px = settings.thigmotaxis_wall_cm * arena.px_per_cm
    near_wall = [d <= wall_px for d in signed_distance(known, arena.floor_px)]
    rles = [f.mask_rle for f in track.frames]
    changes = mask_changes(rles, video.height, video.width)
    return Metrics(
        distance_travelled_cm=distance_travelled_cm(trajectory.points, arena.px_per_cm),
        mean_velocity_cm_s=mean_velocity_cm_s(trajectory.points, arena.px_per_cm, fps),
        centre_time_s=centre.time_s if centre else None,
        centre_entries=centre.entries if centre else None,
        thigmotaxis=sum(near_wall) / len(near_wall) if near_wall else None,
        immobility_s=immobility_s(
            changes, fps, settings.immobility_change_max, settings.immobility_min_s
        ),
        gaps=trajectory.gaps,
        rear_candidates=rear_candidates(
            [f.area_px if f.detected else None for f in track.frames],
            mask_elongations(rles, video.height, video.width),
            speeds_cm_s(trajectory.points, arena.px_per_cm, fps),
            fps,
            settings,
        ),
    )


def _per_frame(points: list[Point | None], known_values: list[float]) -> list[float | None]:
    """Values computed for the known points, put back in frame order with None for the rest."""
    values = iter(known_values)
    return [None if p is None else next(values) for p in points]
