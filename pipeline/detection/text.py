"""
Text detector — PaddleOCR (DBNet), detection-only mode.

Why detection-only (rec=False):
  We need to know WHERE text is, not WHAT it says.
  Skipping recognition (CRNN) is 3-5x faster with no loss of useful information.

Why PaddleOCR over EasyOCR / Tesseract:
  - 80+ language support (assignment requires multilingual scripts)
  - DBNet detector is faster than CRAFT at comparable accuracy
  - Well-maintained, production-used by Baidu at scale

Position heuristic — why we only flag overlay-likely text:
  Brand watermarks, lower-thirds, and social media text stickers appear at
  frame edges: top 20%, bottom 30%, or within 10% of left/right margins.
  Mid-frame text (whiteboards, books, street signs) is not a brand overlay.
  This heuristic cuts false positives significantly at near-zero recall cost
  for actual brand text. Documented as a known limitation in the model card.

PaddleOCR runs on CPU — PaddlePaddle GPU support on MPS is patchy.
On T4/A100 (Kaggle/Docker), set use_gpu=True in _load_model().
Requires paddleocr<3.0.0 — the 3.x API removed use_gpu and ocr(rec=False).
"""

from __future__ import annotations

import concurrent.futures
import numpy as np

from pipeline.config import PipelineConfig
from pipeline.schemas import Detection
from pipeline.detection.base import BaseDetector


class TextDetector(BaseDetector):
    """
    PaddleOCR-based text region detector.
    Returns text bounding boxes as (x1,y1,x2,y2) axis-aligned rects
    converted from DBNet's quadrilateral polygon output.
    """

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._ocr = None

    def _load_model(self) -> None:
        from paddleocr import PaddleOCR

        print("  Loading text model (PaddleOCR DBNet)...")

        # use_gpu=False: stable on both M1 and Linux without GPU paddle install.
        # On Kaggle with paddlepaddle-gpu installed, set use_gpu=True for ~3x speedup.
        self._ocr = PaddleOCR(
            use_angle_cls=False,   # skip text angle classification — not needed for detection
            lang="en",             # base language model; DBNet itself is language-agnostic
            use_gpu=False,
            show_log=False,        # suppress paddle internal logs
            det_db_thresh=0.3,     # DBNet binarization threshold — lower = more recall
            det_db_box_thresh=self.config.text_confidence,
        )
        self._loaded = True

    def detect(self, image: np.ndarray, frame_idx: int) -> list[Detection]:
        """
        Detect text regions in a BGR frame.
        Applies overlay-position heuristic to filter mid-frame text.
        """
        self._ensure_loaded()

        h, w = image.shape[:2]

        # PaddleOCR expects RGB — convert from OpenCV's BGR
        rgb = image[:, :, ::-1]

        try:
            # rec=False: detection only, no character recognition
            # cls=False: skip angle classification
            # 8-second timeout guards against PaddleOCR hanging on specific
            # frame patterns — a known issue on CPU inference for long videos.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(self._ocr.ocr, rgb, rec=False, cls=False)
                result = future.result(timeout=8)
        except concurrent.futures.TimeoutError:
            return []  # skip frame if OCR hangs
        except Exception:
            return []  # graceful degradation on OCR errors

        if not result or result[0] is None:
            return []

        detections: list[Detection] = []

        for item in result[0]:
            # item format: [polygon_points, None]
            # polygon_points: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (may be rotated quad)
            polygon = item[0]

            bbox = _polygon_to_bbox(polygon)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox

            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            if not _is_overlay_position(x1, y1, x2, y2, w, h):
                continue

            detections.append(Detection(
                frame_idx=frame_idx,
                bbox=(x1, y1, x2, y2),
                confidence=self.config.text_confidence,  # DBNet gives box-level conf
                class_name="text",
            ))

        return detections


# ── Helpers ───────────────────────────────────────────────────────────────────

def _polygon_to_bbox(polygon: list) -> tuple[int, int, int, int] | None:
    """
    Convert a quadrilateral polygon to an axis-aligned bounding box.

    PaddleOCR outputs rotated quads to handle angled text. For redaction,
    we only need the enclosing axis-aligned rectangle — rotation precision
    isn't required when we're blurring/filling the region anyway.
    """
    try:
        pts = np.array(polygon, dtype=np.float32)
        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        x2 = int(pts[:, 0].max())
        y2 = int(pts[:, 1].max())
        return x1, y1, x2, y2
    except (ValueError, IndexError, TypeError):
        return None


def _is_overlay_position(
    x1: int, y1: int, x2: int, y2: int,
    frame_w: int, frame_h: int,
) -> bool:
    """
    Return True if this text box is in an overlay-likely position.

    Overlay zones (empirically derived from broadcast + social media content):
      - Top band:    top 20% of frame height  (news tickers, channel logos)
      - Bottom band: bottom 30% of frame height (lower-thirds, subtitles, text stickers)
      - Left/right margins: within 8% of frame width (watermarks, timestamps)

    Mid-frame text (whiteboards, book pages, street signs) is excluded.
    This is a heuristic, not a classifier — documented as a known limitation.
    """
    box_cy = (y1 + y2) / 2  # vertical centre of the text box

    in_top_band    = box_cy < frame_h * 0.20
    in_bottom_band = box_cy > frame_h * 0.70
    in_left_margin = x2 < frame_w * 0.08
    in_right_margin = x1 > frame_w * 0.92

    return in_top_band or in_bottom_band or in_left_margin or in_right_margin
