"""
Redaction engine: apply anonymization to each tracked region.

Per-class strategy:
  face  → Gaussian blur (σ=20, kernel=51) — defeats ArcFace re-ID at σ≥15
  text  → Luminance-matched solid fill — blends with surrounding scene tone
  logo  → Gaussian blur (σ=15, kernel=41) — reads as "out of focus" on surfaces

All redactions use a feathered mask edge (default 3px) to avoid hard rectangular
boundaries that degrade SSIM in surrounding pixels.

Redaction is applied in-place on the frame image to avoid copying large arrays.
The caller is responsible for passing a copy if the original must be preserved
(e.g., in the evaluation pipeline for SSIM comparison).
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline.config import PipelineConfig
from pipeline.schemas import Track


class Redactor:
    """
    Applies class-appropriate anonymization to each active track's bbox.

    Usage:
        redactor = Redactor(config)
        redacted_frame = redactor.redact(frame.image, frame.active_tracks)
        # frame.image is modified in-place; return value is the same array.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def redact(self, image: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """
        Apply redaction for all active tracks. Modifies image in-place.

        Tracks are processed in order — if two tracks overlap (e.g., a logo
        on a person's clothing near their face), both redactions are applied
        sequentially. This is correct: both regions should be anonymized.

        Returns the same image array (modified in-place) for pipeline chaining.
        """
        h, w = image.shape[:2]

        for track in tracks:
            x1, y1, x2, y2 = _clamp_bbox(track.bbox, w, h)

            if x2 <= x1 or y2 <= y1:
                continue  # degenerate bbox after clamping — skip

            if track.class_name == "face":
                _blur_region(
                    image, x1, y1, x2, y2,
                    self.config.face_blur_kernel,
                    self.config.face_blur_sigma,
                    self.config.mask_feather_px,
                )
            elif track.class_name == "text":
                _fill_region(
                    image, x1, y1, x2, y2,
                    self.config.mask_feather_px,
                )
            elif track.class_name == "logo":
                _blur_region(
                    image, x1, y1, x2, y2,
                    self.config.logo_blur_kernel,
                    self.config.logo_blur_sigma,
                    self.config.mask_feather_px,
                )

        return image


# ── Redaction primitives ──────────────────────────────────────────────────────

def _blur_region(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    kernel_size: int,
    sigma: float,
    feather_px: int,
) -> None:
    """
    Gaussian blur a rectangular region in-place.

    Kernel size is capped to the ROI dimensions (must be ≤ min(roi_h, roi_w))
    and always kept odd. Without this cap, OpenCV raises an error when the
    kernel is larger than the image region — which happens on small detections.

    Feathering: the blurred ROI is blended into the original using a soft mask,
    eliminating the hard rectangular boundary that would otherwise degrade SSIM
    in the surrounding pixels.
    """
    roi_h = y2 - y1
    roi_w = x2 - x1

    # Clamp kernel to ROI size — must be odd and ≥1
    k = min(kernel_size, roi_h, roi_w)
    k = k if k % 2 == 1 else k - 1
    k = max(k, 1)

    roi = image[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (k, k), sigma)

    if feather_px > 0 and roi_h > feather_px * 2 and roi_w > feather_px * 2:
        _blend_with_feather(image, x1, y1, x2, y2, blurred, feather_px)
    else:
        image[y1:y2, x1:x2] = blurred


def _fill_region(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    feather_px: int,
) -> None:
    """
    Fill a rectangular region with a luminance-matched solid colour in-place.

    Luminance matching: sample the 5px border outside the text box, compute
    the mean luminance (Y channel in YCrCb space), fill with a grey of that
    luminance. Result reads as "blank space" rather than "black censor bar."

    If the border region is inaccessible (box at frame edge), falls back to
    mid-grey (128, 128, 128).
    """
    fill_bgr = _sample_border_luminance(image, x1, y1, x2, y2, border_px=5)
    filled = np.full((y2 - y1, x2 - x1, 3), fill_bgr, dtype=np.uint8)

    roi_h = y2 - y1
    roi_w = x2 - x1

    if feather_px > 0 and roi_h > feather_px * 2 and roi_w > feather_px * 2:
        _blend_with_feather(image, x1, y1, x2, y2, filled, feather_px)
    else:
        image[y1:y2, x1:x2] = filled


def _blend_with_feather(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    replacement: np.ndarray,
    feather_px: int,
) -> None:
    """
    Blend a replacement region into the image with a soft feathered edge.

    Creates a float32 mask of ones (same size as the ROI), applies Gaussian
    blur to that mask (which creates a gradient near the edges), then uses
    the mask to linearly blend the replacement with the original.

    Result: pixels at the exact boundary are a 50/50 blend; pixels 3px inside
    are >95% replacement; pixels 3px outside are <5% replacement. Smooth.
    """
    roi_h = y2 - y1
    roi_w = x2 - x1

    # Build hard mask, then blur it to create the feather gradient
    mask = np.ones((roi_h, roi_w), dtype=np.float32)
    k = feather_px * 2 + 1  # kernel size for feather blur (always odd)
    mask = cv2.GaussianBlur(mask, (k, k), feather_px / 2.0)
    mask = mask[:, :, np.newaxis]  # (H, W, 1) for broadcasting over 3 channels

    roi = image[y1:y2, x1:x2].astype(np.float32)
    replacement_f = replacement.astype(np.float32)

    blended = mask * replacement_f + (1.0 - mask) * roi
    image[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp_bbox(
    bbox: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """Clamp bbox to valid frame coordinates. Handles expanded boxes at edges."""
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1),
        max(0, y1),
        min(frame_w, x2),
        min(frame_h, y2),
    )


def _sample_border_luminance(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    border_px: int = 5,
) -> tuple[int, int, int]:
    """
    Sample the mean luminance of the border region surrounding a bbox.
    Returns a BGR tuple of (luma, luma, luma) — a neutral grey at that brightness.

    Why YCrCb for luminance: YCrCb separates luma (Y) from chrominance (Cr, Cb).
    The Y channel directly represents perceived brightness, independent of hue.
    Using mean BGR directly would mix chroma into the fill colour, producing
    tinted fills on colourful backgrounds. A neutral grey at the right brightness
    is always less conspicuous than a tinted fill.
    """
    h, w = image.shape[:2]
    samples: list[np.ndarray] = []

    regions = [
        image[max(0, y1 - border_px): y1,          x1:x2],          # top
        image[y2:                      min(h, y2 + border_px), x1:x2],  # bottom
        image[y1:y2,                   max(0, x1 - border_px): x1],  # left
        image[y1:y2,                   x2:min(w, x2 + border_px)],  # right
    ]

    for region in regions:
        if region.size > 0:
            samples.append(region.reshape(-1, 3))

    if not samples:
        return (128, 128, 128)  # mid-grey fallback when box is at frame corner

    all_pixels = np.concatenate(samples, axis=0).astype(np.uint8)

    # Compute mean BGR then convert to YCrCb to extract luma
    mean_bgr = all_pixels.mean(axis=0).astype(np.uint8).reshape(1, 1, 3)
    ycrcb = cv2.cvtColor(mean_bgr, cv2.COLOR_BGR2YCrCb)
    luma = int(ycrcb[0, 0, 0])

    return (luma, luma, luma)
