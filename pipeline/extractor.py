"""
Frame extraction: decode video to FrameData objects via a memory-efficient generator.

Design decisions:
- Generator pattern: constant memory regardless of video length (2 frames in RAM at once).
- OpenCV VideoCapture for sequential decode — simpler than PyAV, sufficient for our use.
- Scene cut detection via grayscale mean-absolute-difference (fast, interpretable).
- max_frames parameter supports short test runs during development.
- resize_for_inference() is a utility for detectors that need a resolution cap.

Production upgrade path:
- Replace cv2.VideoCapture with PyAV for hardware-accelerated decode and PTS access.
- Add a background decode thread with a queue to overlap I/O with GPU inference.
  Pattern: decode thread → frame queue → main inference thread.
  This hides ~30% of decode latency behind compute on fast GPUs.
"""

from __future__ import annotations

from typing import Generator, Optional

import cv2
import numpy as np

from pipeline.config import PipelineConfig
from pipeline.schemas import FrameData, VideoMetadata


def frame_generator(
    meta: VideoMetadata,
    config: PipelineConfig,
    max_frames: Optional[int] = None,
) -> Generator[FrameData, None, None]:
    """
    Yield FrameData objects sequentially from a video file.

    Memory model: O(1) — only two consecutive frames held in RAM at once
    (current frame + previous grayscale for scene cut comparison).

    Args:
        meta:       VideoMetadata from ingestion stage.
        config:     PipelineConfig (scene_cut_threshold used here).
        max_frames: If set, stop after this many frames. Useful for dev/testing.

    Yields:
        FrameData with image (HWC BGR uint8), frame_idx, timestamp_ms,
        is_scene_cut. Fields detections and active_tracks start empty —
        downstream stages populate them.
    """
    cap = cv2.VideoCapture(meta.local_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video file: {meta.local_path}\n"
            "Check the file exists and is a valid video format."
        )

    fps = meta.fps if meta.fps > 0 else 30.0
    prev_gray: Optional[np.ndarray] = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of video

            if max_frames is not None and frame_idx >= max_frames:
                break

            timestamp_ms = (frame_idx / fps) * 1000.0
            is_scene_cut = _detect_scene_cut(frame, prev_gray, config)

            # Keep a grayscale copy of this frame for the next iteration's comparison.
            # Converting to grayscale here (not storing full BGR) saves ~3x memory.
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            yield FrameData(
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                image=frame,          # HWC BGR uint8 — OpenCV native format
                is_scene_cut=is_scene_cut,
            )

            frame_idx += 1

    finally:
        # Always release the capture handle, even if an exception propagates up.
        cap.release()


def _detect_scene_cut(
    frame: np.ndarray,
    prev_gray: Optional[np.ndarray],
    config: PipelineConfig,
) -> bool:
    """
    Detect a hard scene cut by comparing the current frame to the previous one.

    Method: mean absolute difference (MAD) on grayscale images.
    MAD ∈ [0, 255]; values above threshold indicate a sudden visual change.

    Why grayscale: color changes (lighting shift, tint) can inflate the diff
    in color space without being a true scene cut. Grayscale is more robust.

    Why MAD over SSIM or histogram: MAD is O(n) and runs in <0.5ms on 1080p.
    SSIM is more accurate but ~20ms. For scene cut detection accuracy vs. speed,
    MAD is the industry-standard first-pass approach.
    """
    if prev_gray is None:
        return False  # no previous frame — first frame is never a cut

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # float32 prevents uint8 underflow when computing absolute difference
    diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
    return diff > config.scene_cut_threshold


def resize_for_inference(
    image: np.ndarray,
    max_dimension: int,
) -> tuple[np.ndarray, float]:
    """
    Resize an image so its longest dimension does not exceed max_dimension.
    Preserves aspect ratio. Returns (resized_image, scale_factor).

    scale_factor is the ratio of output size to input size.
    Multiply output bounding boxes by (1 / scale_factor) to get original coords.

    Usage: logo and text detectors call this to cap resolution for throughput.
    Face detector skips this — small faces require full resolution.

    Example:
        resized, scale = resize_for_inference(frame, 1280)
        detections = logo_detector.detect(resized)
        # scale boxes back to original frame coordinates:
        boxes = [(int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale))
                 for x1, y1, x2, y2 in boxes]
    """
    h, w = image.shape[:2]
    longest = max(h, w)

    if longest <= max_dimension:
        return image, 1.0

    scale = max_dimension / longest
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def total_frame_count(meta: VideoMetadata) -> int:
    """
    Best-effort total frame count for progress reporting.
    OpenCV's CAP_PROP_FRAME_COUNT is unreliable for some container formats
    (VFR video, certain WebM files). Use as an estimate only.
    """
    cap = cv2.VideoCapture(meta.local_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(count, 0)
