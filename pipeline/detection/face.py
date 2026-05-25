"""
Face detector — YOLOv8n-face.

Model: yolov8n-face.pt, trained on WIDER FACE dataset.
Why WIDER FACE: 393k annotated faces across extreme scale (8x8px min),
pose (profile, tilted), occlusion, and lighting — exactly our target distribution.

Runs at FULL resolution (not downscaled) because small-face recall (≥8×8px
minimum per assignment) degrades sharply with downscaling. A 16×16 face at
1080p becomes an ~5×5 face at 640p — invisible to any detector.

Model source: github.com/akanametov/yolov8-face (AGPL-3.0)
Downloaded once to config.models_dir on first use.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.schemas import Detection
from pipeline.detection.base import BaseDetector

_MODEL_FILENAME = "yolov8n-face.pt"
# Repo renamed: yolov8-face → yolo-face; release tag: v0.0.0 → 1.0.0
_MODEL_URL = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)


class FaceDetector(BaseDetector):
    """
    YOLOv8n-face wrapper. Detects human faces in a single BGR frame.

    Output: list[Detection] with class_name="face", bbox in (x1,y1,x2,y2)
    absolute pixel coordinates of the original input image.
    """

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._model = None

    def _load_model(self) -> None:
        from ultralytics import YOLO

        model_path = self.config.models_dir / _MODEL_FILENAME

        if not model_path.exists():
            print(f"  Downloading face model → {model_path}")
            _download_model(_MODEL_URL, model_path)

        print(f"  Loading face model from {model_path} on {self.config.device}")
        self._model = YOLO(str(model_path))

        # Move model to target device.
        # Ultralytics handles cuda/cpu; MPS is set via the predict() device arg.
        self._loaded = True

    def detect(self, image: np.ndarray, frame_idx: int) -> list[Detection]:
        """
        Run face detection on a single BGR frame.

        No resizing applied — full resolution preserves small-face recall.
        Ultralytics internally resizes to its inference stride (32px aligned)
        and maps boxes back to original coordinates automatically.
        """
        self._ensure_loaded()

        device = str(self.config.device)

        # verbose=False suppresses ultralytics per-frame console output
        results = self._model(
            image,
            conf=self.config.face_confidence,
            device=device,
            verbose=False,
            imgsz=max(image.shape[:2]),  # preserve aspect, don't force square crop
        )

        return _parse_ultralytics_results(results, frame_idx, class_name="face")


# ── Shared ultralytics result parser ─────────────────────────────────────────
# Logo detector (YOLO-World) uses the same output format, so this lives here
# and is imported there. Avoids duplicating the parsing logic.

def _parse_ultralytics_results(
    results,
    frame_idx: int,
    class_name: str,
) -> list[Detection]:
    """
    Convert ultralytics Results object → list[Detection].

    Ultralytics returns boxes in xyxy format as a CUDA/MPS/CPU tensor.
    We call .cpu().numpy() explicitly at the model boundary — this is where
    device tensors become NumPy arrays and stay NumPy for the rest of the pipeline.
    """
    detections: list[Detection] = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        # .xyxy: (N, 4) tensor — x1, y1, x2, y2 in original image pixels
        # .conf: (N,) tensor — confidence scores
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            detections.append(Detection(
                frame_idx=frame_idx,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(conf),
                class_name=class_name,
            ))

    return detections


# ── Model download utility ────────────────────────────────────────────────────

def _download_model(url: str, dest: Path) -> None:
    """Download a model file with a simple progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r    {pct:.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()  # newline after progress
    except Exception as exc:
        dest.unlink(missing_ok=True)  # clean up partial download
        raise RuntimeError(
            f"Failed to download face model from {url}.\n"
            f"Manual fix: download yolov8n-face.pt and place it in {dest.parent}/\n"
            f"Source: https://github.com/akanametov/yolov8-face/releases\n"
            f"Error: {exc}"
        ) from exc
