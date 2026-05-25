"""
Logo / brand mark detector — YOLO-World (open-vocabulary).

Why open-vocabulary is required:
  A fixed-class logo detector fails on any logo outside its training set.
  YOLO-World uses CLIP text-image alignment to understand "logo" as a semantic
  concept, generalising to unseen brands without retraining.

Why YOLO-World over GroundingDINO:
  GroundingDINO: ~300ms/frame on T4 → 3 FPS before tracking/redaction overhead.
  YOLO-World-S:  ~15ms/frame on T4 → throughput-compatible with sparse inference.
  Recall tradeoff: YOLO-World is slightly lower recall on abstract shapes
  (logos without text context). Mitigated by multiple text prompts + low threshold.

Why multiple text prompts:
  "Logo" alone covers obvious branded graphics. Multiple prompts
  ("brand mark", "trademark", "watermark") cover abstract shapes and
  semi-transparent overlays. Results are merged; duplicates removed by IoU.

Model: yolov8s-worldv2.pt — downloaded automatically by ultralytics on first use.
"""

from __future__ import annotations

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.schemas import Detection
from pipeline.detection.base import BaseDetector
from pipeline.detection.face import _parse_ultralytics_results
from pipeline.extractor import resize_for_inference


class LogoDetector(BaseDetector):
    """
    YOLO-World open-vocabulary logo and brand mark detector.
    Queries the image with multiple text prompts and merges results.
    """

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._model = None

    def _load_model(self) -> None:
        from ultralytics import YOLO

        print("  Loading logo model (YOLO-World-S)...")
        # yolov8s-worldv2.pt downloads automatically from ultralytics GitHub releases
        self._model = YOLO("yolov8s-worldv2.pt")

        # Pre-set the text class prompts. set_classes() must be called before
        # predict() — it encodes the prompts into text embeddings via CLIP.
        self._model.set_classes(self.config.logo_prompts)

        print(f"  Logo prompts: {self.config.logo_prompts}")
        self._loaded = True

    def detect(self, image: np.ndarray, frame_idx: int) -> list[Detection]:
        """
        Detect logos and brand marks in a BGR frame.

        Logo detection runs on a resolution-capped image for throughput —
        logos are generally large enough to survive downscaling, unlike faces.
        Boxes are mapped back to original frame coordinates before returning.
        """
        self._ensure_loaded()

        # Cap resolution for inference — logos are larger than faces,
        # so downscaling to max_inference_dimension doesn't hurt recall much.
        resized, scale = resize_for_inference(
            image, self.config.max_inference_dimension
        )

        device = str(self.config.device)

        results = self._model(
            resized,
            conf=self.config.logo_confidence,
            device=device,
            verbose=False,
        )

        detections = _parse_ultralytics_results(results, frame_idx, class_name="logo")

        # Scale bounding boxes back to original frame coordinates.
        # resize_for_inference shrinks the image; detections are in shrunken space.
        if scale < 1.0:
            detections = [_scale_detection(d, scale) for d in detections]

        # Remove duplicates from overlapping prompt matches (IoU > 0.5)
        detections = _nms(detections, iou_threshold=0.5)

        return detections


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scale_detection(det: Detection, scale: float) -> Detection:
    """Map a detection's bbox from resized-image space to original-image space."""
    x1, y1, x2, y2 = det.bbox
    return Detection(
        frame_idx=det.frame_idx,
        bbox=(
            int(x1 / scale),
            int(y1 / scale),
            int(x2 / scale),
            int(y2 / scale),
        ),
        confidence=det.confidence,
        class_name=det.class_name,
    )


def _nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """
    Simple greedy NMS to remove duplicate boxes from overlapping prompt matches.
    Keeps the higher-confidence box when two boxes have IoU > threshold.

    This is not the same as the NMS inside YOLO-World — that removes duplicate
    detections per prompt. This removes duplicates *across* prompts (e.g., the
    same logo matched by both "logo" and "brand mark" prompts).
    """
    if len(detections) <= 1:
        return detections

    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []

    for det in detections:
        if all(_iou(det.bbox, k.bbox) < iou_threshold for k in kept):
            kept.append(det)

    return kept


def _iou(a: tuple, b: tuple) -> float:
    """Intersection over Union for two (x1,y1,x2,y2) bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0
