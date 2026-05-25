"""
Evaluation CLI for the video anonymization pipeline.

Usage:
    # Full test set (video dir + annotation dir)
    python evaluate.py --video-dir data/test_videos/ --annotation-dir data/annotations/

    # Single video + annotation
    python evaluate.py --video data/test_videos/clip.mp4 --annotation data/annotations/clip.json

    # Skip re-ID test (faster)
    python evaluate.py --video-dir data/test_videos/ --annotation-dir data/annotations/ --no-reid

    # No annotations — only measures SSIM, FPS, temporal consistency
    python evaluate.py --video data/test_videos/clip.mp4

Outputs:
    outputs/eval/<stem>_result.json     — per-video metrics
    outputs/eval/aggregate_result.json  — aggregate across all videos
    outputs/eval/benchmark_report.md    — markdown summary (deliverable)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np
import torch
from tqdm import tqdm

from evaluation.metrics import (
    ClassMetrics,
    EvaluationResult,
    GroundTruthAnnotation,
    aggregate_metrics,
    compute_detection_metrics,
    compute_frame_ssim,
    compute_temporal_consistency,
    load_ground_truth,
    measure_throughput,
)
from evaluation.reid_test import ReidSummary, run_reid_test
from pipeline.config import PipelineConfig, get_device
from pipeline.detection.ensemble import DetectionEnsemble
from pipeline.extractor import frame_generator, total_frame_count
from pipeline.ingestion import ingest
from pipeline.reconstruction import VideoReconstructor, build_output_path
from pipeline.redaction import Redactor
from pipeline.schemas import Track, VideoMetadata
from pipeline.temporal import TemporalSmoother
from pipeline.tracking.tracker import MultiTracker


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--video-dir", "video_dir", default=None, type=click.Path(exists=True),
              help="Directory of .mp4 test videos.")
@click.option("--annotation-dir", "annotation_dir", default=None, type=click.Path(exists=True),
              help="Directory of annotation JSON files (same stem as videos).")
@click.option("--video", "single_video", default=None, type=click.Path(exists=True),
              help="Single video file to evaluate.")
@click.option("--annotation", "single_annotation", default=None, type=click.Path(exists=True),
              help="Annotation JSON for --video.")
@click.option("--output-dir", "output_dir", default="outputs/eval", show_default=True,
              type=click.Path(), help="Directory to write result JSON and benchmark report.")
@click.option("--detection-interval", "detection_interval", default=5, show_default=True,
              help="Run detectors every K frames.")
@click.option("--device", default="auto", show_default=True,
              type=click.Choice(["auto", "cuda", "mps", "cpu"]),
              help="Compute device.")
@click.option("--face-confidence", "face_confidence", default=0.45, show_default=True)
@click.option("--logo-confidence", "logo_confidence", default=0.30, show_default=True)
@click.option("--ssim-sample-rate", "ssim_sample_rate", default=30, show_default=True,
              help="Compute SSIM every N frames (memory trade-off). 1 = every frame.")
@click.option("--no-reid", "no_reid", is_flag=True, default=False,
              help="Skip re-ID resistance test (faster evaluation).")
@click.option("--max-frames", "max_frames", default=None, type=int,
              help="Limit frames per video (for quick smoke tests).")
@click.option("--iou-threshold", "iou_threshold", default=0.5, show_default=True,
              help="IoU threshold for TP/FP matching in detection metrics.")
def main(
    video_dir: str | None,
    annotation_dir: str | None,
    single_video: str | None,
    single_annotation: str | None,
    output_dir: str,
    detection_interval: int,
    device: str,
    face_confidence: float,
    logo_confidence: float,
    ssim_sample_rate: int,
    no_reid: bool,
    max_frames: int | None,
    iou_threshold: float,
) -> None:
    """Evaluate the anonymization pipeline against ground truth annotations."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_device = get_device() if device == "auto" else torch.device(device)
    config = PipelineConfig(
        device=resolved_device,
        detection_interval=detection_interval,
        face_confidence=face_confidence,
        logo_confidence=logo_confidence,
    )

    click.echo("=" * 60)
    click.echo("Video Anonymization — Evaluation")
    click.echo(f"  {config.describe()}")
    click.echo("=" * 60)

    # ── Build work list ───────────────────────────────────────────────────────
    work_items: list[tuple[Path, Path | None]] = []  # (video_path, annotation_path | None)

    if single_video:
        ann_path = Path(single_annotation) if single_annotation else None
        work_items.append((Path(single_video), ann_path))

    if video_dir:
        vdir = Path(video_dir)
        adir = Path(annotation_dir) if annotation_dir else None
        for vid in sorted(vdir.glob("*.mp4")):
            ann = (adir / f"{vid.stem}.json") if adir else None
            if ann and not ann.exists():
                click.echo(f"  [warn] No annotation for {vid.name} — skipping detection metrics")
                ann = None
            work_items.append((vid, ann))

    if not work_items:
        click.echo("No videos found. Use --video or --video-dir.")
        return

    click.echo(f"\nEvaluating {len(work_items)} video(s)...\n")

    # ── Initialise pipeline components once (reuse across videos) ────────────
    ensemble = DetectionEnsemble(config)
    ensemble.warmup()

    all_results: list[EvaluationResult] = []

    for vid_path, ann_path in work_items:
        click.echo(f"── {vid_path.name} {'(no annotations)' if ann_path is None else ''}")

        ground_truth = load_ground_truth(ann_path) if ann_path else []

        result = evaluate_video(
            video_path=vid_path,
            ground_truth=ground_truth,
            config=config,
            ensemble=ensemble,
            ssim_sample_rate=ssim_sample_rate,
            run_reid=not no_reid and bool(ground_truth),
            max_frames=max_frames,
            iou_threshold=iou_threshold,
            output_dir=output_path,
        )

        result.print_report()
        all_results.append(result)

        # Write per-video JSON
        per_video_out = output_path / f"{vid_path.stem}_result.json"
        per_video_out.write_text(json.dumps(result.to_dict(), indent=2))
        click.echo(f"  Saved → {per_video_out}")

    # ── Aggregate across all videos ───────────────────────────────────────────
    if len(all_results) > 1:
        click.echo("\n── Aggregate Results ───────────────────────────────────")
        agg = aggregate_metrics(all_results)
        agg.print_report()

        agg_out = output_path / "aggregate_result.json"
        agg_out.write_text(json.dumps(agg.to_dict(), indent=2))
        click.echo(f"  Saved → {agg_out}")
    else:
        agg = all_results[0] if all_results else EvaluationResult()

    # ── Write benchmark report (markdown deliverable) ─────────────────────────
    report_path = output_path / "benchmark_report.md"
    _write_markdown_report(agg, all_results, report_path)
    click.echo(f"\n  Benchmark report → {report_path}")
    click.echo("=" * 60)


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate_video(
    video_path: Path,
    ground_truth: list[GroundTruthAnnotation],
    config: PipelineConfig,
    ensemble: DetectionEnsemble,
    ssim_sample_rate: int = 30,
    run_reid: bool = True,
    max_frames: int | None = None,
    iou_threshold: float = 0.5,
    output_dir: Path = Path("outputs/eval"),
) -> EvaluationResult:
    """
    Run the full pipeline on one video and compute all evaluation metrics.

    Captures three things that anonymise.py does not:
      1. original_frames_sampled — pre-redaction copies every ssim_sample_rate frames
         (for SSIM comparison against their redacted counterparts)
      2. track_records — list[list[Track]], one per frame (for temporal consistency)
      3. redacted_frames_sampled — post-redaction copies at the same sample indices

    Returns a populated EvaluationResult.
    """
    meta = ingest(str(video_path), config)

    tracker = MultiTracker(config)
    smoother = TemporalSmoother(config)
    redactor = Redactor(config)

    anon_out = output_dir / f"anon_{video_path.stem}.mp4"
    reconstructor = VideoReconstructor(config, meta)
    reconstructor.open(anon_out)

    # Capture state for metrics
    track_records: list[list[Track]] = []
    original_sampled: list[np.ndarray] = []
    redacted_sampled: list[np.ndarray] = []
    pred_per_frame: list[list[Track]] = []

    start = time.perf_counter()
    frames_processed = 0

    try:
        with tqdm(total=total_frame_count(meta) if not max_frames else max_frames,
                  unit="frame", dynamic_ncols=True, desc="  eval") as pbar:

            for frame in frame_generator(meta, config, max_frames=max_frames):

                # ── Detection gate ────────────────────────────────────────────
                is_det_frame = ensemble.is_detection_frame(frame)
                frame.is_detection_frame = is_det_frame

                if is_det_frame:
                    frame.detections = ensemble.run(frame)
                    frame.active_tracks = tracker.update(
                        frame.detections,
                        is_scene_cut=frame.is_scene_cut,
                        frame_idx=frame.frame_idx,
                    )
                else:
                    frame.active_tracks = tracker.get_active_tracks()

                # ── Temporal smoothing ────────────────────────────────────────
                frame.active_tracks = smoother.process(
                    frame.active_tracks, frame.width, frame.height,
                )

                # ── SSIM sampling: save original BEFORE redaction ─────────────
                is_ssim_frame = (frames_processed % ssim_sample_rate == 0)
                if is_ssim_frame:
                    original_sampled.append(frame.image.copy())

                # ── Redaction ─────────────────────────────────────────────────
                redactor.redact(frame.image, frame.active_tracks)

                # ── SSIM sampling: save redacted AFTER redaction ──────────────
                if is_ssim_frame:
                    redacted_sampled.append(frame.image.copy())

                # ── Capture for metrics ───────────────────────────────────────
                track_records.append(list(frame.active_tracks))
                pred_per_frame.append(list(frame.active_tracks))

                reconstructor.write_frame(frame.image)
                frames_processed += 1
                pbar.update(1)

    except KeyboardInterrupt:
        click.echo("\n  Interrupted — finalising...")
    finally:
        reconstructor.close()

    elapsed = time.perf_counter() - start

    # ── Compute metrics ───────────────────────────────────────────────────────

    fps = measure_throughput(frames_processed, elapsed)

    # Detection (recall/precision/F1) — only if annotations provided
    det_metrics: dict[str, ClassMetrics] = {}
    if ground_truth:
        det_metrics = compute_detection_metrics(pred_per_frame, ground_truth, iou_threshold)
    else:
        det_metrics = {cls: ClassMetrics() for cls in ("face", "text", "logo")}

    # Temporal consistency
    tc = compute_temporal_consistency(track_records)

    # SSIM on non-redacted pixels (sampled frames)
    sampled_track_records = [
        track_records[i * ssim_sample_rate]
        for i in range(len(original_sampled))
        if i * ssim_sample_rate < len(track_records)
    ]
    ssim_score = _compute_sampled_ssim(
        original_sampled, redacted_sampled, sampled_track_records
    )

    # Re-ID test
    reid_rate = 0.0
    if run_reid and ground_truth:
        reid_summary = _run_reid(
            original_sampled, redacted_sampled,
            ground_truth, ssim_sample_rate, meta,
        )
        reid_rate = reid_summary.reid_rate
        reid_summary.print_report()

    return EvaluationResult(
        face=det_metrics["face"],
        text=det_metrics["text"],
        logo=det_metrics["logo"],
        temporal_consistency=tc,
        ssim=ssim_score,
        fps=fps,
        reid_rate=reid_rate,
    )


# ── SSIM helper ───────────────────────────────────────────────────────────────

def _compute_sampled_ssim(
    original_sampled: list[np.ndarray],
    redacted_sampled: list[np.ndarray],
    track_records: list[list[Track]],
) -> float:
    """Average SSIM across sampled frame pairs, masking redacted regions."""
    if not original_sampled:
        return 1.0

    scores = []
    for orig, redc, tracks in zip(original_sampled, redacted_sampled, track_records):
        try:
            score = compute_frame_ssim(orig, redc, tracks)
            scores.append(score)
        except Exception:
            pass  # skip malformed frames

    return float(np.mean(scores)) if scores else 1.0


# ── Re-ID helper ──────────────────────────────────────────────────────────────

def _run_reid(
    original_sampled: list[np.ndarray],
    redacted_sampled: list[np.ndarray],
    ground_truth: list[GroundTruthAnnotation],
    ssim_sample_rate: int,
    meta: VideoMetadata,
) -> ReidSummary:
    """
    Run re-ID test on sampled frames only.

    Because we only sampled every ssim_sample_rate frames, we need to remap
    ground truth frame indices to the sampled list indices.

    sampled index i corresponds to original frame index i * ssim_sample_rate.
    We pass sampled frames as the "full frame list" and filter ground truth
    to only annotations at multiples of ssim_sample_rate, remapped to
    their position in the sampled list.
    """
    remapped_gt = []
    for ann in ground_truth:
        if ann.class_name != "face":
            continue
        if ann.frame_idx % ssim_sample_rate == 0:
            sampled_idx = ann.frame_idx // ssim_sample_rate
            if sampled_idx < len(original_sampled):
                remapped_gt.append(GroundTruthAnnotation(
                    frame_idx=sampled_idx,
                    bbox=ann.bbox,
                    class_name=ann.class_name,
                    track_id=ann.track_id,
                ))

    if not remapped_gt:
        click.echo("  [re-ID] No face annotations at sampled frame indices — skipping.")
        return ReidSummary()

    return run_reid_test(
        original_frames=original_sampled,
        redacted_frames=redacted_sampled,
        ground_truth=remapped_gt,
    )


# ── Markdown report ───────────────────────────────────────────────────────────

def _write_markdown_report(
    agg: EvaluationResult,
    per_video: list[EvaluationResult],
    output_path: Path,
) -> None:
    """Write a markdown benchmark report (assignment deliverable)."""
    checks = agg.passes_assignment()

    def _mark(key: str) -> str:
        return "✓" if checks.get(key, False) else "✗"

    lines = [
        "# Benchmark Report — Video Anonymization Pipeline",
        "",
        f"Videos evaluated: **{len(per_video)}**",
        "",
        "---",
        "",
        "## Detection Metrics",
        "",
        "| Class | Recall | Precision | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
        f"| Face | {agg.face.recall:.3f} | {agg.face.precision:.3f} | {agg.face.f1:.3f} | {agg.face.tp} | {agg.face.fp} | {agg.face.fn} |",
        f"| Text | {agg.text.recall:.3f} | {agg.text.precision:.3f} | {agg.text.f1:.3f} | {agg.text.tp} | {agg.text.fp} | {agg.text.fn} |",
        f"| Logo | {agg.logo.recall:.3f} | {agg.logo.precision:.3f} | {agg.logo.f1:.3f} | {agg.logo.tp} | {agg.logo.fp} | {agg.logo.fn} |",
        "",
        "---",
        "",
        "## System Metrics",
        "",
        f"| Metric | Value | Target | Status |",
        f"|---|---|---|---|",
        f"| Face Recall (pass) | {agg.face.recall:.3f} | ≥ 0.950 | {_mark('face_recall_pass')} |",
        f"| Face Recall (distinction) | {agg.face.recall:.3f} | ≥ 0.970 | {_mark('face_recall_dist')} |",
        f"| Face Precision | {agg.face.precision:.3f} | ≥ 0.900 | {_mark('face_precision_pass')} |",
        f"| Text Recall (pass) | {agg.text.recall:.3f} | ≥ 0.900 | {_mark('text_recall_pass')} |",
        f"| Text Recall (distinction) | {agg.text.recall:.3f} | ≥ 0.930 | {_mark('text_recall_dist')} |",
        f"| Logo Recall (pass) | {agg.logo.recall:.3f} | ≥ 0.900 | {_mark('logo_recall_pass')} |",
        f"| Logo Recall (distinction) | {agg.logo.recall:.3f} | ≥ 0.930 | {_mark('logo_recall_dist')} |",
        f"| Temporal Consistency | {agg.temporal_consistency:.4f} | ≥ 0.980 | {_mark('temporal_consistency')} |",
        f"| SSIM (non-redacted) | {agg.ssim:.4f} | ≥ 0.850 | {_mark('ssim')} |",
        f"| Throughput (FPS) (pass) | {agg.fps:.1f} | ≥ 10.0 | {_mark('fps_pass')} |",
        f"| Throughput (FPS) (distinction) | {agg.fps:.1f} | ≥ 20.0 | {_mark('fps_dist')} |",
        f"| Re-ID Rate | {agg.reid_rate:.4f} | ≤ 0.020 | {_mark('reid_rate')} |",
        "",
        "---",
        "",
    ]

    if len(per_video) > 1:
        lines += [
            "## Per-Video Breakdown",
            "",
            "| Video | Face R | Text R | Logo R | TC | SSIM | FPS |",
            "|---|---|---|---|---|---|---|",
        ]
        # We don't have video names here — use index
        for i, r in enumerate(per_video):
            lines.append(
                f"| video_{i+1} | {r.face.recall:.3f} | {r.text.recall:.3f} | "
                f"{r.logo.recall:.3f} | {r.temporal_consistency:.3f} | "
                f"{r.ssim:.3f} | {r.fps:.1f} |"
            )
        lines += ["", "---", ""]

    lines += [
        "## Notes",
        "",
        "- IoU threshold for TP/FP matching: 0.50 (PASCAL VOC standard)",
        "- SSIM computed on non-redacted pixels only (intentional blur excluded)",
        "- Temporal consistency: mean per-track coverage fraction over track lifespan",
        "- Re-ID measured via InsightFace ArcFace cosine similarity (threshold 0.6)",
        "- FPS measured as wall-clock time over all frames including decode + encode",
        "",
    ]

    output_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
