"""
Evaluation metrics for the video anonymization pipeline.

Five metrics measured:

1. Detection metrics (per class): recall, precision, F1
   - IoU-based matching between predicted bboxes and ground truth annotations
   - PASCAL VOC greedy matching: sort predictions by confidence descending,
     match each to its highest-IoU unmatched ground truth

2. Temporal consistency
   - Per-track: what fraction of frames within the track's lifespan had a box?
   - Averaged across all tracks
   - Requires track_records: a list (one entry per processed frame) of list[Track]

3. SSIM on non-redacted regions
   - Compute full per-pixel SSIM map between original and redacted frames
   - Average SSIM only in pixels that were NOT inside any redacted bbox
   - This isolates codec/processing degradation from intentional redaction blur

4. FPS throughput
   - Wall-clock frames_processed / elapsed_seconds

5. Re-ID rate (see reid_test.py)
   - Fraction of redacted face regions where ArcFace cosine similarity
     with original face embedding still exceeds matching threshold (0.6)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline.schemas import Track


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ClassMetrics:
    """Detection quality for a single class (face, text, or logo)."""
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    # FPR = false_alarm_frames / negative_frames
    # "negative frame" = frame where this class has no GT boxes
    # "false alarm" = negative frame where the pipeline fired at least one detection
    fpr: float = 0.0

    def __str__(self) -> str:
        return (
            f"recall={self.recall:.3f}  precision={self.precision:.3f}  "
            f"F1={self.f1:.3f}  FPR={self.fpr:.3f}  "
            f"(TP={self.tp} FP={self.fp} FN={self.fn})"
        )


@dataclass
class EvaluationResult:
    """Aggregated evaluation result across a full video."""
    face: ClassMetrics = field(default_factory=ClassMetrics)
    text: ClassMetrics = field(default_factory=ClassMetrics)
    logo: ClassMetrics = field(default_factory=ClassMetrics)
    temporal_consistency: float = 0.0
    ssim: float = 0.0
    fps: float = 0.0
    reid_rate: float = 0.0       # filled in by reid_test.py if run

    def passes_assignment(self) -> dict[str, bool]:
        """Check which assignment thresholds are met."""
        # FPR threshold applies across all three categories combined
        combined_fpr = (self.face.fpr + self.text.fpr + self.logo.fpr) / 3
        return {
            "face_recall_pass":      self.face.recall >= 0.95,
            "face_recall_dist":      self.face.recall >= 0.97,
            "face_precision_pass":   self.face.precision >= 0.90,
            "text_recall_pass":      self.text.recall >= 0.90,
            "text_recall_dist":      self.text.recall >= 0.93,
            "logo_recall_pass":      self.logo.recall >= 0.90,
            "logo_recall_dist":      self.logo.recall >= 0.93,
            "false_positive_rate":   combined_fpr <= 0.05,
            "temporal_consistency":  self.temporal_consistency >= 0.98,
            "ssim":                  self.ssim >= 0.85,
            "fps_pass":              self.fps >= 10.0,
            "fps_dist":              self.fps >= 20.0,
            "reid_rate":             self.reid_rate <= 0.02,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    def print_report(self) -> None:
        checks = self.passes_assignment()
        print("\n── Evaluation Report ──────────────────────────────")
        combined_fpr = (self.face.fpr + self.text.fpr + self.logo.fpr) / 3
        print(f"  Face:  {self.face}")
        print(f"  Text:  {self.text}")
        print(f"  Logo:  {self.logo}")
        print(f"  FPR (combined mean): {combined_fpr:.4f}  (target ≤ 0.05)")
        print(f"  Temporal consistency: {self.temporal_consistency:.4f}")
        print(f"  SSIM (non-redacted):  {self.ssim:.4f}")
        print(f"  Throughput:           {self.fps:.1f} FPS")
        if self.reid_rate > 0:
            print(f"  Re-ID rate:           {self.reid_rate:.4f}")
        print("\n── Pass/Fail ───────────────────────────────────────")
        for key, passed in checks.items():
            mark = "✓" if passed else "✗"
            print(f"  [{mark}] {key}")
        print("────────────────────────────────────────────────────\n")


# ── Ground truth ──────────────────────────────────────────────────────────────

@dataclass
class GroundTruthAnnotation:
    """One ground truth box in one frame."""
    frame_idx: int
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) absolute pixels
    class_name: str                    # "face", "text", or "logo"
    track_id: Optional[int] = None


def load_ground_truth(annotation_path: str | Path) -> list[GroundTruthAnnotation]:
    """
    Load ground truth annotations from a JSON file.

    Expected format:
    {
        "annotations": [
            {"frame_idx": 0, "bbox": [x1, y1, x2, y2], "class": "face", "track_id": 1},
            ...
        ]
    }

    bbox values are absolute pixels matching the original video resolution.
    class must be one of: "face", "text", "logo".
    track_id is optional — used only for temporal consistency ground truth.
    """
    path = Path(annotation_path)
    with path.open() as f:
        data = json.load(f)

    annotations = []
    for entry in data.get("annotations", []):
        bbox = tuple(entry["bbox"])
        annotations.append(GroundTruthAnnotation(
            frame_idx=entry["frame_idx"],
            bbox=bbox,
            class_name=entry["class"],
            track_id=entry.get("track_id"),
        ))
    return annotations


# ── Detection metrics ─────────────────────────────────────────────────────────

def compute_detection_metrics(
    predictions: list[list[Track]],
    ground_truth: list[GroundTruthAnnotation],
    iou_threshold: float = 0.5,
) -> dict[str, ClassMetrics]:
    """
    Compute recall, precision, and F1 for each class via IoU-based matching.

    Args:
        predictions: one list[Track] per frame (index = frame_idx)
        ground_truth: flat list of GroundTruthAnnotation objects
        iou_threshold: minimum IoU to count as a true positive

    Returns:
        dict with keys "face", "text", "logo" mapping to ClassMetrics

    Algorithm (PASCAL VOC greedy):
      For each frame:
        - Sort predictions by confidence descending
        - For each prediction, find the highest-IoU unmatched GT box of the same class
        - If IoU ≥ threshold → TP; mark GT box as matched
        - If no match → FP
      Any GT box not matched → FN
    """
    # Index GT by frame_idx for O(1) lookup
    gt_by_frame: dict[int, list[GroundTruthAnnotation]] = {}
    for ann in ground_truth:
        gt_by_frame.setdefault(ann.frame_idx, []).append(ann)

    # Accumulators per class
    tp = {"face": 0, "text": 0, "logo": 0}
    fp = {"face": 0, "text": 0, "logo": 0}
    fn = {"face": 0, "text": 0, "logo": 0}

    for frame_idx, frame_tracks in enumerate(predictions):
        gt_anns = gt_by_frame.get(frame_idx, [])

        # Separate by class
        for cls in ("face", "text", "logo"):
            preds_cls = [t for t in frame_tracks if t.class_name == cls]
            gt_cls = [a for a in gt_anns if a.class_name == cls]

            matched_gt = set()  # indices into gt_cls that have been consumed

            # Sort by confidence descending — high-confidence predictions get
            # first pick at GT boxes (PASCAL VOC convention)
            preds_cls_sorted = sorted(preds_cls, key=lambda t: t.confidence, reverse=True)

            for pred in preds_cls_sorted:
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gt_cls):
                    if gt_idx in matched_gt:
                        continue
                    iou = _iou(pred.bbox, gt.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold:
                    tp[cls] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp[cls] += 1

            # Every unmatched GT box is a false negative
            fn[cls] += len(gt_cls) - len(matched_gt)

    # FPR: count frames where class is absent (no GT) but pipeline fired a detection
    false_alarm_frames = {cls: 0 for cls in ("face", "text", "logo")}
    negative_frames    = {cls: 0 for cls in ("face", "text", "logo")}

    for frame_idx, frame_tracks in enumerate(predictions):
        gt_anns = gt_by_frame.get(frame_idx, [])
        for cls in ("face", "text", "logo"):
            gt_cls   = [a for a in gt_anns if a.class_name == cls]
            pred_cls = [t for t in frame_tracks if t.class_name == cls]
            if len(gt_cls) == 0:                  # negative frame for this class
                negative_frames[cls] += 1
                if len(pred_cls) > 0:             # pipeline fired on a negative frame
                    false_alarm_frames[cls] += 1

    result = {}
    for cls in ("face", "text", "logo"):
        t = tp[cls]
        p = fp[cls]
        n = fn[cls]
        recall    = t / (t + n) if (t + n) > 0 else 0.0
        precision = t / (t + p) if (t + p) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        fpr = (
            false_alarm_frames[cls] / negative_frames[cls]
            if negative_frames[cls] > 0 else 0.0
        )
        result[cls] = ClassMetrics(
            recall=recall, precision=precision, f1=f1,
            tp=t, fp=p, fn=n, fpr=fpr,
        )

    return result


# ── Temporal consistency ──────────────────────────────────────────────────────

def compute_temporal_consistency(track_records: list[list[Track]]) -> float:
    """
    Measure what fraction of frames within each track's lifespan had a bbox.

    A track's lifespan = [first_frame_seen, last_frame_seen].
    Consistency for that track = frames_with_box / lifespan_length.
    Final score = mean across all tracks.

    A score of 1.0 means every track was present in every frame of its lifespan
    (no single-frame gaps). This is what ByteTrack + Kalman propagation achieves
    when working correctly.

    Args:
        track_records: list with one entry per frame; each entry is the list of
                       active Track objects for that frame (from frame.active_tracks)

    Returns:
        float in [0, 1], or 1.0 if no tracks were observed (vacuously true)
    """
    # For each (class, track_id) pair: record which frames it appeared in
    track_frame_sets: dict[tuple[str, int], set[int]] = {}

    for frame_idx, tracks in enumerate(track_records):
        for track in tracks:
            key = (track.class_name, track.track_id)
            track_frame_sets.setdefault(key, set()).add(frame_idx)

    if not track_frame_sets:
        return 1.0

    consistencies = []
    for key, frames_seen in track_frame_sets.items():
        first = min(frames_seen)
        last  = max(frames_seen)
        lifespan = last - first + 1
        consistency = len(frames_seen) / lifespan
        consistencies.append(consistency)

    return float(np.mean(consistencies))


# ── SSIM on non-redacted regions ──────────────────────────────────────────────

def compute_frame_ssim(
    original: np.ndarray,
    redacted: np.ndarray,
    tracks: list[Track],
) -> float:
    """
    Compute SSIM between original and redacted frames, averaged only over
    pixels that are NOT inside any redacted bounding box.

    Why exclude redacted regions: SSIM should measure pipeline-introduced
    degradation (compression, colour shift), not the intentional privacy blur.
    Including redacted pixels would tank SSIM even for a perfect pipeline.

    Uses skimage structural_similarity with full=True to get a per-pixel map,
    then masks out redacted pixels before computing the mean.

    Args:
        original: BGR uint8 frame (H, W, 3) before redaction
        redacted:  BGR uint8 frame (H, W, 3) after redaction
        tracks:    active tracks for this frame (defines which regions were redacted)

    Returns:
        float in [-1, 1]; 1.0 = perfect reconstruction in non-redacted areas.
        Returns 1.0 if the entire frame was redacted (no unmasked pixels).
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError as e:
        raise ImportError("scikit-image is required for SSIM. pip install scikit-image") from e

    import cv2

    h, w = original.shape[:2]

    # Build a binary mask: 1 = pixel is NOT redacted, 0 = inside a bbox
    mask = np.ones((h, w), dtype=np.float32)
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0.0

    # Convert both frames to greyscale for SSIM (faster + matches standard practice)
    orig_grey = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    redc_grey = cv2.cvtColor(redacted, cv2.COLOR_BGR2GRAY)

    # full=True returns (score, ssim_map) where ssim_map has same spatial dims
    _, ssim_map = structural_similarity(
        orig_grey, redc_grey, full=True, data_range=255
    )

    # Apply mask: average only non-redacted pixels
    non_redacted_pixels = mask > 0.5
    if not non_redacted_pixels.any():
        return 1.0  # entire frame redacted — metric undefined, treat as passing

    return float(ssim_map[non_redacted_pixels].mean())


def compute_video_ssim(
    original_frames: list[np.ndarray],
    redacted_frames: list[np.ndarray],
    track_records: list[list[Track]],
) -> float:
    """
    Average compute_frame_ssim across all frames in a video.

    original_frames and redacted_frames must be the same length.
    track_records[i] contains the active tracks for frame i.
    """
    if not original_frames:
        return 1.0

    scores = [
        compute_frame_ssim(orig, redc, tracks)
        for orig, redc, tracks in zip(original_frames, redacted_frames, track_records)
    ]
    return float(np.mean(scores))


# ── Throughput ────────────────────────────────────────────────────────────────

def measure_throughput(frames_processed: int, elapsed_seconds: float) -> float:
    """
    Compute frames per second.

    Args:
        frames_processed: total number of frames written to output
        elapsed_seconds: wall-clock time from first frame decode to final encode

    Returns:
        FPS as a float. Returns 0.0 if elapsed_seconds is 0.
    """
    if elapsed_seconds <= 0:
        return 0.0
    return frames_processed / elapsed_seconds


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """
    Intersection over Union for two axis-aligned bboxes (x1, y1, x2, y2).
    Returns 0.0 if boxes do not overlap.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def aggregate_metrics(
    per_video_results: list[EvaluationResult],
) -> EvaluationResult:
    """
    Macro-average EvaluationResult across multiple videos.

    Each video's TP/FP/FN counts are summed before computing recall/precision/F1
    (micro-average for detection metrics). Temporal consistency, SSIM, FPS, and
    re-ID rate are simple means across videos.

    Micro-averaging is the right choice here: a video with 500 faces should
    contribute more to the face recall estimate than one with 5 faces. Simple
    mean would give them equal weight.
    """
    if not per_video_results:
        return EvaluationResult()

    # Sum TP/FP/FN per class (micro)
    total = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in ("face", "text", "logo")}
    for r in per_video_results:
        for cls, metrics_obj in [("face", r.face), ("text", r.text), ("logo", r.logo)]:
            total[cls]["tp"] += metrics_obj.tp
            total[cls]["fp"] += metrics_obj.fp
            total[cls]["fn"] += metrics_obj.fn

    def _make_metrics(cls: str) -> ClassMetrics:
        t = total[cls]["tp"]
        p = total[cls]["fp"]
        n = total[cls]["fn"]
        recall    = t / (t + n) if (t + n) > 0 else 0.0
        precision = t / (t + p) if (t + p) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        # FPR: micro-average across videos (mean of per-video FPRs)
        fpr = float(np.mean([getattr(r, cls).fpr for r in per_video_results]))
        return ClassMetrics(recall=recall, precision=precision, f1=f1, tp=t, fp=p, fn=n, fpr=fpr)

    return EvaluationResult(
        face=_make_metrics("face"),
        text=_make_metrics("text"),
        logo=_make_metrics("logo"),
        temporal_consistency=float(np.mean([r.temporal_consistency for r in per_video_results])),
        ssim=float(np.mean([r.ssim for r in per_video_results])),
        fps=float(np.mean([r.fps for r in per_video_results])),
        reid_rate=float(np.mean([r.reid_rate for r in per_video_results])),
    )
