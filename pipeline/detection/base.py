"""
Abstract base class for all detectors (face, text, logo).

All detectors implement the same interface so the ensemble can call them
interchangeably. This is the Strategy pattern — the pipeline never depends
on a specific detector implementation, only on this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.schemas import Detection


class BaseDetector(ABC):
    """
    Interface every detector must implement.

    Lifecycle:
        detector = FaceDetector(config)   # no model loaded yet
        detector.warmup()                 # loads model + runs dummy inference
        detections = detector.detect(img, frame_idx=0)  # fast from here on
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._loaded: bool = False

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _load_model(self) -> None:
        """
        Load model weights into memory (and GPU if applicable).
        Called once, lazily, on the first detect() call or explicit warmup().
        """

    @abstractmethod
    def detect(self, image: np.ndarray, frame_idx: int) -> list[Detection]:
        """
        Run inference on a single HWC BGR uint8 image.

        Args:
            image:     Frame in OpenCV format (HWC, BGR, uint8).
            frame_idx: Frame index from the video — stored in each Detection.

        Returns:
            list[Detection] with bbox in (x1, y1, x2, y2) absolute pixel
            coords of the *input image* (not any resized copy).
        """

    # ── Concrete helpers ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load_model()

    def warmup(self) -> None:
        """
        Prime the model with a dummy forward pass.

        CUDA/MPS JIT-compiles kernel graphs on the first call for a given
        input shape — costing 0.5–2s. Warmup moves that cost to startup,
        keeping per-frame latency consistent during the main loop.
        Production inference servers always do this before accepting traffic.
        """
        self._ensure_loaded()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, frame_idx=-1)

    def detect_batch(
        self,
        images: list[np.ndarray],
        frame_indices: list[int],
    ) -> list[list[Detection]]:
        """
        Run detection on multiple images.
        Default loops over detect(). Subclasses can override for native
        batch inference, which is more GPU-efficient than serial calls.
        """
        return [
            self.detect(img, idx)
            for img, idx in zip(images, frame_indices)
        ]
