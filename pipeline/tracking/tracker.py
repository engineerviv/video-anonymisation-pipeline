"""
MultiTracker: per-class ByteTrack instances with scene-cut reset.

Design decisions:
- One ByteTrack instance per detection class (face/text/logo).
  Prevents cross-class track ID collisions and enables per-class threshold tuning.

- Detection frames: call ByteTrack update → store results in _active_tracks.
- Non-detection frames: return _active_tracks (last known positions, no tracker call).
  Rationale: supervision's ByteTrack behaves inconsistently with empty-detection
  calls across versions. Maintaining our own state is version-stable and explicit.

- Scene cuts: reset all trackers + clear state.
  A scene cut invalidates all Kalman motion predictions — stale tracks on new
  scene content create false positives. Full reset is the correct response.

Upgrade paths:
- For faster motion content: call tracker on every frame (drop sparse inference).
- For better identity continuity across long occlusions: swap ByteTrack for
  DeepSORT (adds appearance embedding model, ~10ms overhead per frame).
- For multi-camera consistency: StrongSORT or BoT-SORT with re-ID features.
"""

from __future__ import annotations

import warnings
import numpy as np
import supervision as sv

from pipeline.config import PipelineConfig
from pipeline.schemas import Detection, Track


class MultiTracker:
    """
    Manages one ByteTrack instance per detection class.

    Usage pattern (inside main pipeline loop):
        tracker = MultiTracker(config)

        for frame in frames:
            if frame.is_detection_frame:
                tracks = tracker.update(frame.detections, frame.is_scene_cut, frame.frame_idx)
            else:
                tracks = tracker.get_active_tracks()

            frame.active_tracks = tracks
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._trackers: dict[str, sv.ByteTrack] = {}
        # (class_name, track_id) → Track — last confirmed positions
        self._active_tracks: dict[tuple[str, int], Track] = {}
        # (class_name, track_id) → frame_idx when track first appeared
        self._birth_frames: dict[tuple[str, int], int] = {}
        self._init_trackers()

    # ── Public interface ──────────────────────────────────────────────────────

    def update(
        self,
        detections: list[Detection],
        is_scene_cut: bool,
        frame_idx: int,
    ) -> list[Track]:
        """
        Process detections from a detection frame.
        Resets tracker state if this frame is a scene cut.
        Updates _active_tracks with newly confirmed tracks.

        Args:
            detections:   All detections from this frame (all classes mixed).
            is_scene_cut: If True, reset all trackers before updating.
            frame_idx:    Current frame index (used to compute track age).

        Returns:
            list[Track] — confirmed tracks with updated positions.
        """
        if is_scene_cut:
            self._reset()

        # Group detections by class — each class has its own tracker
        by_class: dict[str, list[Detection]] = {"face": [], "text": [], "logo": []}
        for det in detections:
            by_class[det.class_name].append(det)

        new_active: dict[tuple[str, int], Track] = {}

        for class_name, class_dets in by_class.items():
            sv_dets = _detections_to_sv(class_dets)
            tracked = self._trackers[class_name].update_with_detections(sv_dets)

            for track in _sv_to_tracks(tracked, class_name, frame_idx, self._birth_frames):
                key = (class_name, track.track_id)
                new_active[key] = track

        self._active_tracks = new_active
        return list(new_active.values())

    def get_active_tracks(self) -> list[Track]:
        """
        Return last known track positions for non-detection frames.
        No tracker call — boxes are held at their last detected positions.
        The temporal smoother handles visual continuity between detections.
        """
        return list(self._active_tracks.values())

    def reset(self) -> None:
        """Public reset — call explicitly on pipeline restart or test teardown."""
        self._reset()

    @property
    def track_count(self) -> int:
        return len(self._active_tracks)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_trackers(self) -> None:
        """Create fresh ByteTrack instances for all three classes."""
        for class_name in ("face", "text", "logo"):
            self._trackers[class_name] = _make_bytetrack(class_name, self.config)

    def _reset(self) -> None:
        """Full state reset — called on scene cuts and explicit resets."""
        self._active_tracks.clear()
        self._birth_frames.clear()
        self._init_trackers()


# ── ByteTrack factory ─────────────────────────────────────────────────────────

def _make_bytetrack(class_name: str, config: PipelineConfig) -> sv.ByteTrack:
    """
    Create a ByteTrack instance tuned for the given detection class.

    Per-class threshold rationale:
    - face: higher activation threshold (0.45) — we trust face detector confidence.
    - text: moderate (0.50) — DBNet box-level confidence is well-calibrated.
    - logo: lower (0.30) — we intentionally run logo detection at low confidence
            for recall, so ByteTrack should accept these lower-confidence detections.

    minimum_consecutive_frames = track_min_hits:
        How many consecutive detection matches before a track is "confirmed" and
        returned to the pipeline. Higher = fewer spurious tracks, slightly lower
        recall on short-lived objects.

    lost_track_buffer = track_max_age:
        Frames a track can survive without a detection match. Must be > K
        (detection_interval) or tracks will die between detection frames.
        Default 30 >> K=5, so tracks survive the sparse inference gaps safely.
    """
    threshold = {
        "face": config.face_confidence,
        "text": config.text_confidence,
        "logo": config.logo_confidence,
    }[class_name]

    # Suppress FutureWarning: ByteTrack deprecated in supervision 0.28, removed in 0.30.
    # requirements.txt pins supervision<0.30 until we migrate to the new API.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return sv.ByteTrack(
            track_activation_threshold=threshold,
            lost_track_buffer=config.track_max_age,
            minimum_matching_threshold=0.8,
            minimum_consecutive_frames=config.track_min_hits,
        )


# ── Schema conversion ─────────────────────────────────────────────────────────

def _detections_to_sv(detections: list[Detection]) -> sv.Detections:
    """
    Convert our Detection list to supervision's Detections format.
    Empty list → sv.Detections.empty() (supervision's null object pattern).
    """
    if not detections:
        return sv.Detections.empty()

    xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
    conf = np.array([d.confidence for d in detections], dtype=np.float32)

    return sv.Detections(xyxy=xyxy, confidence=conf)


def _sv_to_tracks(
    tracked: sv.Detections,
    class_name: str,
    frame_idx: int,
    birth_frames: dict[tuple[str, int], int],
) -> list[Track]:
    """
    Convert supervision's tracked Detections → our Track list.

    tracker_id is None if no tracks exist (ByteTrack returns None for empty scenes).
    We guard against this and return an empty list safely.

    Track age is computed from birth_frames dict. We register the birth frame
    on first appearance and compute age as (current - birth + 1).
    """
    if tracked.tracker_id is None or len(tracked) == 0:
        return []

    tracks: list[Track] = []

    for i in range(len(tracked)):
        track_id = int(tracked.tracker_id[i])
        key = (class_name, track_id)

        # Register birth frame on first appearance
        if key not in birth_frames:
            birth_frames[key] = frame_idx

        age = frame_idx - birth_frames[key] + 1
        x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
        conf = (
            float(tracked.confidence[i])
            if tracked.confidence is not None
            else 0.5
        )

        tracks.append(Track(
            track_id=track_id,
            class_name=class_name,
            bbox=(x1, y1, x2, y2),
            confidence=conf,
            age=age,
            frames_since_update=0,   # supervision only returns currently-matched tracks
            is_confirmed=True,        # ByteTrack only returns confirmed tracks
        ))

    return tracks
