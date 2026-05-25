from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

# All bbox coordinates are (x1, y1, x2, y2) in absolute pixels — never
# normalized [0,1] or (x,y,w,h). Conversions happen at each model boundary.
BBox = tuple[int, int, int, int]
DetectionClass = Literal["face", "text", "logo"]


@dataclass
class Detection:
    """
    One bounding box from one detector on one frame.
    Immutable after creation — detectors produce these, they don't modify them.
    """
    frame_idx: int
    bbox: BBox                    # (x1, y1, x2, y2) absolute pixels
    confidence: float             # [0.0, 1.0]
    class_name: DetectionClass
    track_id: Optional[int] = None  # assigned by tracker; None before tracking


@dataclass
class Track:
    """
    One object being tracked by ByteTrack across frames.
    Populated by the tracker from Kalman predictions or detector updates.
    """
    track_id: int
    class_name: DetectionClass
    bbox: BBox                    # Kalman-predicted or last-updated position
    confidence: float
    age: int                      # total frames since track was initialized
    frames_since_update: int      # frames since last detector match (0 = just updated)
    is_confirmed: bool            # True once track has been matched min_hits times


@dataclass
class VideoMetadata:
    """
    Extracted from the downloaded video before processing begins.
    Drives frame extraction settings (FPS, resolution, audio mux).
    """
    url: str
    local_path: str
    fps: float
    width: int
    height: int
    duration_seconds: float
    has_audio: bool
    codec: str
    filesize_bytes: int


@dataclass
class FrameData:
    """
    The unit of work that flows through the entire pipeline.

    Lifecycle:
      extractor   → creates with image, frame_idx, timestamp_ms, is_scene_cut
      ensemble    → populates detections (on detection frames only)
      tracker     → populates active_tracks (every frame, via predict or update)
      temporal    → smooths active_tracks in-place
      redactor    → modifies image in-place
      reconstructor → reads image and writes to output stream
    """
    frame_idx: int
    timestamp_ms: float
    image: np.ndarray             # HWC, BGR, uint8 — matches OpenCV convention
    detections: list[Detection] = field(default_factory=list)
    active_tracks: list[Track] = field(default_factory=list)
    is_scene_cut: bool = False
    is_detection_frame: bool = False

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]
