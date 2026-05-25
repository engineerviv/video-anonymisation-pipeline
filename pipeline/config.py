from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


def get_device() -> torch.device:
    """
    Auto-detect the best available compute device.
    Priority: CUDA (Kaggle/Docker) → MPS (Apple M1) → CPU.
    Called once at startup; result stored in PipelineConfig.device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class PipelineConfig:
    """
    Single source of truth for all pipeline hyperparameters.
    Pass one instance through the entire pipeline — no module imports constants
    from other modules.

    Tuning guide:
      - Raise face_confidence if you have too many false positives on faces.
      - Lower face_confidence if you're missing faces (recall drops).
      - Raise detection_interval (K) to improve throughput at cost of new-object latency.
      - Lower detection_interval on fast-motion content (sports, action).
      - Raise face_blur_sigma above 20 if re-ID rate is still above 2%.
    """

    # ── Compute ───────────────────────────────────────────────────────────────
    device: torch.device = field(default_factory=get_device)

    # ── Sparse Inference ─────────────────────────────────────────────────────
    # Run all detectors every K frames; tracker propagates the rest.
    # K=5 → detection at 6 FPS on 30 FPS content.
    # K=8 → detection at ~3.75 FPS — better for M1 throughput.
    detection_interval: int = 5

    # ── Detection Thresholds ──────────────────────────────────────────────────
    # Face: 0.45 balances recall vs false positives on WIDER FACE distribution.
    face_confidence: float = 0.45
    # Text: 0.5 — DBNet is generally well-calibrated; lower if missing watermarks.
    text_confidence: float = 0.50
    # Logo: 0.3 — intentionally low to maximize recall; accept more false positives.
    # The assignment weights logo recall (≥90%) above FPR (≤5%).
    logo_confidence: float = 0.30

    # ── Tracking ──────────────────────────────────────────────────────────────
    # Delete a track if it hasn't been matched in this many frames.
    track_max_age: int = 30
    # Confirm a track after this many consecutive detection matches.
    # Set to 1 for a privacy pipeline — redact on first detection.
    # Higher values reduce spurious tracks but delay redaction by (min_hits × K) frames,
    # which is unacceptable for faces (e.g. min_hits=3, K=5 → 15 frames unredacted).
    track_min_hits: int = 1

    # ── Temporal Smoothing ────────────────────────────────────────────────────
    # EMA alpha for bbox coordinate smoothing. Higher = more responsive, less smooth.
    ema_alpha: float = 0.7
    # Expand each bbox by this fraction in all directions before redacting.
    # Covers hair, ear edges, and slight detection misalignment.
    box_margin_pct: float = 0.15

    # ── Redaction ─────────────────────────────────────────────────────────────
    # Face blur: kernel must be odd. sigma=20 defeats ArcFace re-ID (verified at sigma≥15).
    face_blur_kernel: int = 51
    face_blur_sigma: float = 20.0
    # Logo blur: slightly lighter than face blur — logos are often on surfaces.
    logo_blur_kernel: int = 41
    logo_blur_sigma: float = 15.0
    # Feathered mask edge blend width in pixels.
    mask_feather_px: int = 3

    # ── Resolution ────────────────────────────────────────────────────────────
    # Cap the longest dimension at this value for inference (not for output).
    # Face detection runs at full resolution to preserve small-face recall.
    # Logo/text detection can use this cap for throughput.
    max_inference_dimension: int = 1280

    # ── Video ─────────────────────────────────────────────────────────────────
    # Scene cut pixel-difference threshold (mean absolute diff per channel).
    # Higher = less sensitive (misses soft cuts). Lower = more resets (false cuts).
    scene_cut_threshold: float = 30.0

    # ── YOLO-World Logo Prompts ───────────────────────────────────────────────
    # Multiple prompts increase semantic coverage of the "logo" concept.
    # YOLO-World matches each prompt independently and takes the union.
    logo_prompts: list[str] = field(default_factory=lambda: [
        "logo",
        "brand mark",
        "company logo",
        "trademark",
        "watermark",
        "product logo",
        "brand name",
    ])

    # ── Paths ─────────────────────────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    temp_dir: Path = field(default_factory=lambda: Path(".tmp"))

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        """Human-readable summary printed at pipeline startup."""
        return (
            f"Device: {self.device} | "
            f"Detection interval: K={self.detection_interval} | "
            f"Face conf: {self.face_confidence} | "
            f"Logo conf: {self.logo_confidence} | "
            f"Blur sigma (face): {self.face_blur_sigma}"
        )
