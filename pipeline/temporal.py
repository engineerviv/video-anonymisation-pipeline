"""
TemporalSmoother: EMA smoothing, box expansion, and dead-track cleanup.

Sits between the tracker and the redactor. Takes a list of Track objects
from the tracker, returns a list of Track objects with:
  1. EMA-smoothed bbox coordinates (eliminates inter-frame jitter)
  2. Expanded bboxes (covers hair/ears/edges missed by tight detection boxes)
  3. Cleaned state for tracks that have left the scene

Why this is a separate layer from tracking:
  The tracker's job is identity association — which detection belongs to which
  track. The temporal smoother's job is trajectory refinement — making the
  track's spatial signal clean enough for high-quality redaction. Separating
  these concerns keeps each module focused and independently testable.
"""

from __future__ import annotations

from pipeline.config import PipelineConfig
from pipeline.schemas import BBox, Track


class TemporalSmoother:
    """
    Applies EMA smoothing and box expansion to track bounding boxes.

    Maintains per-track EMA state across frames. Cleans up state for tracks
    that are no longer active (prevents unbounded memory growth on long videos).

    One instance per pipeline run — state accumulates across all frames.
    Reset between videos by creating a new instance.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        # (class_name, track_id) → last smoothed bbox
        self._ema_state: dict[tuple[str, int], BBox] = {}

    def process(
        self,
        tracks: list[Track],
        frame_w: int,
        frame_h: int,
    ) -> list[Track]:
        """
        Smooth and expand all active track bboxes.

        Args:
            tracks:   Active tracks from MultiTracker for this frame.
            frame_w:  Frame width in pixels — used to clamp expanded boxes.
            frame_h:  Frame height in pixels — used to clamp expanded boxes.

        Returns:
            New list of Track objects with smoothed + expanded bboxes.
            Original Track objects are not mutated.
        """
        active_keys: set[tuple[str, int]] = set()
        result: list[Track] = []

        for track in tracks:
            key = (track.class_name, track.track_id)
            active_keys.add(key)

            smoothed = self._apply_ema(key, track.bbox)
            expanded = _expand_bbox(smoothed, self.config.box_margin_pct, frame_w, frame_h)

            # Return a new Track with updated bbox — don't mutate the input.
            # Dataclasses are mutable by default, but explicit copies
            # make the data flow clear: smoother input ≠ smoother output.
            result.append(Track(
                track_id=track.track_id,
                class_name=track.class_name,
                bbox=expanded,
                confidence=track.confidence,
                age=track.age,
                frames_since_update=track.frames_since_update,
                is_confirmed=track.is_confirmed,
            ))

        # Remove EMA state for tracks that are no longer active.
        # Dead track = was in _ema_state but not in this frame's active set.
        # Without this cleanup, _ema_state grows across the full video timeline.
        dead = set(self._ema_state.keys()) - active_keys
        for key in dead:
            del self._ema_state[key]

        return result

    def _apply_ema(self, key: tuple[str, int], new_bbox: BBox) -> BBox:
        """
        Apply exponential moving average to a bounding box.

        First appearance: store as-is (no prior history to blend).
        Subsequent frames: blend new position with stored history.

        EMA formula: smoothed = α × new + (1-α) × previous
        α = config.ema_alpha (default 0.7)

        Integer rounding: we cast to int after blending. Subpixel precision
        is meaningless for pixel-aligned redaction operations.
        """
        if key not in self._ema_state:
            self._ema_state[key] = new_bbox
            return new_bbox

        alpha = self.config.ema_alpha
        prev = self._ema_state[key]

        smoothed: BBox = (
            int(alpha * new_bbox[0] + (1 - alpha) * prev[0]),
            int(alpha * new_bbox[1] + (1 - alpha) * prev[1]),
            int(alpha * new_bbox[2] + (1 - alpha) * prev[2]),
            int(alpha * new_bbox[3] + (1 - alpha) * prev[3]),
        )

        self._ema_state[key] = smoothed
        return smoothed

    @property
    def active_track_count(self) -> int:
        """Number of tracks currently holding EMA state."""
        return len(self._ema_state)


# ── Bbox expansion ────────────────────────────────────────────────────────────

def _expand_bbox(
    bbox: BBox,
    margin_pct: float,
    frame_w: int,
    frame_h: int,
) -> BBox:
    """
    Expand a bounding box by margin_pct of its own dimensions in all directions.

    Scale-aware: a 200px-wide box expands by 30px per side (15% × 200 / 2).
    An 8px-wide box expands by 0.6px → rounded to 1px per side.
    Fixed-pixel margins would over-expand small boxes and under-expand large ones.

    Result is clamped to [0, frame_w] × [0, frame_h] — never out of bounds.
    """
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1

    # Each side gets half the total percentage expansion
    mx = int(box_w * margin_pct / 2)
    my = int(box_h * margin_pct / 2)

    return (
        max(0, x1 - mx),
        max(0, y1 - my),
        min(frame_w, x2 + mx),
        min(frame_h, y2 + my),
    )
