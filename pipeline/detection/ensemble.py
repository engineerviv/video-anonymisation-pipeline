"""
DetectionEnsemble: orchestrates all three detectors with sparse inference gating.

Responsibilities:
  1. Sparse inference gate — run detectors every K frames, always on scene cuts.
  2. Sequential detector execution with per-detector latency tracking.
  3. Graceful degradation — one detector failure does not abort the frame.
  4. Result merging — returns a single list[Detection] for the tracker.

What this does NOT do:
  - Cross-class NMS: overlapping face+logo boxes (e.g. logo on clothing near face)
    should both be redacted independently — keeping all is correct behaviour.
  - Batching across frames: each call processes one frame's detection pass.
    Batch-across-frames is a valid production optimisation but adds complexity.

Threading note:
  Sequential execution chosen over parallel because:
  - GIL prevents Python-level parallelism for CPU-bound work.
  - Face + logo share the same GPU device; concurrent calls need careful stream management.
  - Text runs on CPU; overlapping CPU+GPU is possible but adds coordination overhead.
  - At K=5 sparse, sequential per-detection-frame cost is amortised across 5 frames.
  Upgrade path: use torch.cuda.Stream per model for true GPU-level overlap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pipeline.config import PipelineConfig
from pipeline.schemas import Detection, FrameData
from pipeline.detection.face import FaceDetector
from pipeline.detection.text import TextDetector
from pipeline.detection.logo import LogoDetector


@dataclass
class LatencyStats:
    """Accumulated per-detector latency for benchmarking."""
    face_ms: float = 0.0
    text_ms: float = 0.0
    logo_ms: float = 0.0
    detection_frames: int = 0

    def average(self) -> dict[str, float]:
        n = max(self.detection_frames, 1)
        return {
            "face_ms":  self.face_ms  / n,
            "text_ms":  self.text_ms  / n,
            "logo_ms":  self.logo_ms  / n,
            "total_ms": (self.face_ms + self.text_ms + self.logo_ms) / n,
        }


class DetectionEnsemble:
    """
    Holds all three detector instances and applies the sparse inference gate.

    Usage:
        ensemble = DetectionEnsemble(config)
        ensemble.warmup()                      # prime all models before main loop
        for frame in frame_generator(...):
            frame.detections = ensemble.run(frame)   # empty list on non-detection frames
            frame.is_detection_frame = ensemble.is_detection_frame(frame)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.face = FaceDetector(config)
        self.text = TextDetector(config)
        self.logo = LogoDetector(config)
        self.stats = LatencyStats()

    def warmup(self) -> None:
        """
        Load all models and prime their GPU/MPS graphs with dummy inference.
        Call this once before the main processing loop.
        Moves model load time + JIT compilation cost to startup.
        """
        print("Warming up detectors...")
        self.face.warmup()
        print("  ✓ Face detector ready")
        self.text.warmup()
        print("  ✓ Text detector ready")
        self.logo.warmup()
        print("  ✓ Logo detector ready")

    def is_detection_frame(self, frame: FrameData) -> bool:
        """
        Return True if this frame should trigger full detection.

        Two conditions trigger detection:
          1. Scheduled: frame_idx is a multiple of detection_interval (K).
          2. Forced:    frame is a scene cut — tracker state is invalid post-cut,
                        so we must re-detect immediately rather than waiting for
                        the next scheduled frame (up to K-1 frames of stale boxes).
        """
        scheduled = (frame.frame_idx % self.config.detection_interval == 0)
        return scheduled or frame.is_scene_cut

    def run(self, frame: FrameData) -> list[Detection]:
        """
        Run all detectors on this frame if the sparse gate passes.
        Returns an empty list on non-detection frames — the tracker handles those.

        Per-detector failures are caught and logged; processing continues with
        results from the surviving detectors. A corrupt frame should never
        abort the pipeline.
        """
        if not self.is_detection_frame(frame):
            return []

        image = frame.image
        idx = frame.frame_idx
        all_detections: list[Detection] = []

        # ── Face detection ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            face_dets = self.face.detect(image, idx)
            all_detections.extend(face_dets)
        except Exception as exc:
            print(f"  [WARN] Face detector failed on frame {idx}: {exc}")
            face_dets = []
        self.stats.face_ms += (time.perf_counter() - t0) * 1000

        # ── Text detection ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            text_dets = self.text.detect(image, idx)
            all_detections.extend(text_dets)
        except Exception as exc:
            print(f"  [WARN] Text detector failed on frame {idx}: {exc}")
            text_dets = []
        self.stats.text_ms += (time.perf_counter() - t0) * 1000

        # ── Logo detection ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            logo_dets = self.logo.detect(image, idx)
            all_detections.extend(logo_dets)
        except Exception as exc:
            print(f"  [WARN] Logo detector failed on frame {idx}: {exc}")
            logo_dets = []
        self.stats.logo_ms += (time.perf_counter() - t0) * 1000

        self.stats.detection_frames += 1

        return all_detections

    def print_latency_report(self) -> None:
        """Print average per-detector latency. Call after processing completes."""
        avg = self.stats.average()
        print("\n── Detection Latency Report ────────────────────────")
        print(f"  Face detector:  {avg['face_ms']:.1f} ms/detection-frame")
        print(f"  Text detector:  {avg['text_ms']:.1f} ms/detection-frame")
        print(f"  Logo detector:  {avg['logo_ms']:.1f} ms/detection-frame")
        print(f"  Total per det.frame: {avg['total_ms']:.1f} ms")
        print(f"  Detection frames processed: {self.stats.detection_frames}")
        k = self.config.detection_interval
        amortised = avg['total_ms'] / k
        print(f"  Amortised over K={k}: ~{amortised:.1f} ms/frame → ~{1000/amortised:.0f} FPS ceiling")
        print("────────────────────────────────────────────────────\n")
