"""
Re-identification resistance test using InsightFace ArcFace.

Question being answered: after Gaussian blur is applied to a face region,
can a face recognition model still identify who that person is?

Protocol:
  1. For each annotated face bbox in the test set, load:
       - The original (pre-redaction) frame
       - The redacted frame produced by the pipeline
  2. Crop both frames to the face bbox, resize to 112×112 (ArcFace input).
  3. Run ArcFace to produce 512-dim L2-normalised embedding vectors.
  4. Compute cosine similarity between original and redacted embeddings.
  5. If similarity ≥ threshold (0.6 = standard ArcFace same-person threshold),
     the face was not successfully anonymized — count as re-identified.

Re-ID rate = re_identified_count / total_faces_tested

Target: ≤ 2%.

Why 0.6: The InsightFace/ArcFace paper shows that cosine similarity ≥ 0.6
between two embeddings means "same person" with very high probability on
standard benchmarks (LFW, CFP-FP, AgeDB-30). Values below 0.6 are "different
person" territory. Gaussian blur at σ=20 typically drops similarity to 0.1–0.3
on faces ≥ 40×40px.

Why no landmark alignment: Proper ArcFace use involves 5-point landmark
detection followed by affine alignment to a canonical face pose. Here we skip
alignment because:
  (a) We already have bboxes from ground truth, not from a face detector.
  (b) The same crop (with the same possible misalignment) is taken from both
      the original and redacted frames — the comparison is self-consistent.
  (c) Skipping alignment slightly reduces absolute embedding accuracy, but
      the relative comparison (original vs. blurred version of the same crop)
      is still valid for measuring re-ID resistance.

Model used: InsightFace buffalo_sc pack — lightweight face recognition model
(MobileFaceNet backbone, w600k_mbf.onnx). Downloaded automatically on first
run to ~/.insightface/models/.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from evaluation.metrics import GroundTruthAnnotation


# ── ArcFace wrapper ───────────────────────────────────────────────────────────

class ArcFaceEmbedder:
    """
    Thin wrapper around InsightFace ArcFace ONNX model.

    Handles lazy loading — model is downloaded on first use only.
    Runs on CPU (evaluation path, not in main pipeline).
    Input: BGR uint8 image of any size.
    Output: 512-dim float32 embedding vector (L2-normalised).
    """

    ARCFACE_INPUT_SIZE: int = 112   # ArcFace canonical input: 112×112 px
    MATCH_THRESHOLD: float = 0.6    # cosine similarity ≥ this → same person

    def __init__(self, model_pack: str = "buffalo_sc") -> None:
        """
        Args:
            model_pack: InsightFace model pack name.
                        "buffalo_sc" — small/fast (MobileFaceNet backbone).
                        "buffalo_l"  — large/accurate (ResNet50 backbone).
        """
        self._model_pack = model_pack
        self._rec_model = None   # lazy load

    def _load(self) -> None:
        """Download and load the recognition model on first call."""
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "insightface is required for re-ID evaluation. "
                "pip install insightface onnxruntime"
            ) from e

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            app = FaceAnalysis(
                name=self._model_pack,
                providers=["CPUExecutionProvider"],
            )
            # ctx_id=-1 forces CPU
            app.prepare(ctx_id=-1, det_size=(640, 640))

        # Extract just the recognition sub-model from the app
        if "recognition" not in app.models:
            raise RuntimeError(
                f"Model pack '{self._model_pack}' has no recognition model. "
                "Try 'buffalo_l' or 'buffalo_sc'."
            )
        self._rec_model = app.models["recognition"]

    def embed(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        """
        Produce a 512-dim L2-normalised embedding for a face crop.

        Args:
            face_crop_bgr: BGR uint8 image of the face region (any size).
                           Will be resized to 112×112 internally.

        Returns:
            np.ndarray of shape (512,), dtype float32, L2-normalised.
        """
        if self._rec_model is None:
            self._load()

        # Resize to ArcFace input size (preserves aspect if already square)
        size = self.ARCFACE_INPUT_SIZE
        resized = cv2.resize(face_crop_bgr, (size, size), interpolation=cv2.INTER_LINEAR)

        # get_feat handles BGR→RGB conversion and normalisation internally
        embedding = self._rec_model.get_feat([resized]).flatten()

        # L2-normalise for cosine similarity via dot product
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normalised embeddings.
        Since both are L2-normalised, this reduces to a dot product.
        Range: [-1, 1], higher = more similar.
        """
        return float(np.dot(emb_a, emb_b))

    def is_reidentified(self, emb_original: np.ndarray, emb_redacted: np.ndarray) -> bool:
        """
        Returns True if the redacted face is still identifiable as the original.
        Threshold: 0.6 (ArcFace standard same-person boundary).
        """
        return self.cosine_similarity(emb_original, emb_redacted) >= self.MATCH_THRESHOLD


# ── Per-face result ───────────────────────────────────────────────────────────

@dataclass
class FaceReidResult:
    """Re-ID test result for a single face crop."""
    frame_idx: int
    bbox: tuple[int, int, int, int]
    face_size_px: int              # min(width, height) of the crop
    cosine_similarity: float
    is_reidentified: bool
    # Embeddings stored for post-hoc analysis (optional)
    embedding_original: Optional[np.ndarray] = field(default=None, repr=False)
    embedding_redacted: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class ReidSummary:
    """Aggregate re-ID results across all tested faces."""
    total_faces: int = 0
    reidentified: int = 0
    reid_rate: float = 0.0
    mean_similarity: float = 0.0
    # Breakdown by face size bucket
    small_reid_rate: float = 0.0    # faces < 40px (likely low regardless of blur)
    medium_reid_rate: float = 0.0   # 40px–80px (the hard regime)
    large_reid_rate: float = 0.0    # > 80px (blur should dominate)
    per_face: list[FaceReidResult] = field(default_factory=list)

    def print_report(self) -> None:
        print("\n── Re-ID Resistance Report ────────────────────────")
        print(f"  Total faces tested:   {self.total_faces}")
        print(f"  Re-identified:        {self.reidentified}")
        print(f"  Re-ID rate:           {self.reid_rate:.4f}  (target ≤ 0.02)")
        passed = "✓ PASS" if self.reid_rate <= 0.02 else "✗ FAIL"
        print(f"  Assignment check:     {passed}")
        print(f"  Mean cosine sim:      {self.mean_similarity:.4f}")
        print(f"\n  By face size:")
        print(f"    < 40px:   {self.small_reid_rate:.4f}")
        print(f"    40–80px:  {self.medium_reid_rate:.4f}")
        print(f"    > 80px:   {self.large_reid_rate:.4f}")
        print("────────────────────────────────────────────────────\n")


# ── Main test function ────────────────────────────────────────────────────────

def run_reid_test(
    original_frames: list[np.ndarray],
    redacted_frames: list[np.ndarray],
    ground_truth: list[GroundTruthAnnotation],
    model_pack: str = "buffalo_sc",
    min_face_size: int = 20,
    store_embeddings: bool = False,
) -> ReidSummary:
    """
    Run ArcFace re-ID resistance test across a set of frames.

    Args:
        original_frames: list of BGR frames BEFORE redaction, indexed by frame_idx
        redacted_frames:  list of BGR frames AFTER redaction, same indexing
        ground_truth:     GroundTruthAnnotation list (only "face" class used)
        model_pack:       InsightFace model pack (default: "buffalo_sc")
        min_face_size:    skip faces smaller than this (px) — too small for meaningful test
        store_embeddings: if True, save embedding vectors in each FaceReidResult

    Returns:
        ReidSummary with per-face results and aggregate statistics
    """
    embedder = ArcFaceEmbedder(model_pack=model_pack)

    face_anns = [a for a in ground_truth if a.class_name == "face"]
    per_face: list[FaceReidResult] = []

    for ann in face_anns:
        frame_idx = ann.frame_idx
        if frame_idx >= len(original_frames) or frame_idx >= len(redacted_frames):
            continue

        x1, y1, x2, y2 = ann.bbox
        face_w = x2 - x1
        face_h = y2 - y1
        face_size = min(face_w, face_h)

        if face_size < min_face_size:
            continue  # too small to test meaningfully

        orig_frame = original_frames[frame_idx]
        redc_frame = redacted_frames[frame_idx]

        # Clamp bbox to frame bounds
        h, w = orig_frame.shape[:2]
        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(w, x2); y2c = min(h, y2)

        if x2c <= x1c or y2c <= y1c:
            continue

        orig_crop = orig_frame[y1c:y2c, x1c:x2c]
        redc_crop = redc_frame[y1c:y2c, x1c:x2c]

        emb_orig = embedder.embed(orig_crop)
        emb_redc = embedder.embed(redc_crop)

        sim = embedder.cosine_similarity(emb_orig, emb_redc)
        reid = embedder.is_reidentified(emb_orig, emb_redc)

        per_face.append(FaceReidResult(
            frame_idx=frame_idx,
            bbox=(x1c, y1c, x2c, y2c),
            face_size_px=face_size,
            cosine_similarity=sim,
            is_reidentified=reid,
            embedding_original=emb_orig if store_embeddings else None,
            embedding_redacted=emb_redc if store_embeddings else None,
        ))

    return _summarise(per_face)


def run_reid_test_from_paths(
    original_frame_paths: list[str | Path],
    redacted_frame_paths: list[str | Path],
    ground_truth: list[GroundTruthAnnotation],
    model_pack: str = "buffalo_sc",
    min_face_size: int = 20,
) -> ReidSummary:
    """
    Convenience wrapper: loads frames from disk paths rather than arrays.
    Useful when running evaluation from saved frame images rather than in-memory.

    original_frame_paths[i] is the frame at frame_idx=i (0-indexed).
    """
    def _load(paths: list[str | Path]) -> list[np.ndarray]:
        frames = []
        for p in paths:
            frame = cv2.imread(str(p))
            if frame is None:
                raise ValueError(f"Could not read frame: {p}")
            frames.append(frame)
        return frames

    original_frames = _load(original_frame_paths)
    redacted_frames = _load(redacted_frame_paths)
    return run_reid_test(original_frames, redacted_frames, ground_truth, model_pack, min_face_size)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _summarise(per_face: list[FaceReidResult]) -> ReidSummary:
    """Compute aggregate statistics from per-face results."""
    if not per_face:
        return ReidSummary(total_faces=0, reid_rate=0.0)

    total = len(per_face)
    n_reid = sum(1 for r in per_face if r.is_reidentified)
    mean_sim = float(np.mean([r.cosine_similarity for r in per_face]))
    reid_rate = n_reid / total

    def _bucket_rate(results: list[FaceReidResult], lo: int, hi: int) -> float:
        bucket = [r for r in results if lo <= r.face_size_px < hi]
        if not bucket:
            return 0.0
        return sum(1 for r in bucket if r.is_reidentified) / len(bucket)

    return ReidSummary(
        total_faces=total,
        reidentified=n_reid,
        reid_rate=reid_rate,
        mean_similarity=mean_sim,
        small_reid_rate=_bucket_rate(per_face, 0, 40),
        medium_reid_rate=_bucket_rate(per_face, 40, 80),
        large_reid_rate=_bucket_rate(per_face, 80, 10_000),
        per_face=per_face,
    )


def embed_single_face(
    face_crop_bgr: np.ndarray,
    model_pack: str = "buffalo_sc",
) -> np.ndarray:
    """
    Utility: get ArcFace embedding for a single face crop.
    Useful for quick one-off checks outside the full evaluation loop.
    """
    return ArcFaceEmbedder(model_pack=model_pack).embed(face_crop_bgr)
