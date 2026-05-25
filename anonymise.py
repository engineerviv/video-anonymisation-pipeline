"""
Video Anonymization Pipeline — CLI entrypoint.

Usage:
    python anonymise.py --url "https://www.youtube.com/watch?v=<id>"
    python anonymise.py --url /path/to/local/clip.mp4 --max-frames 150
    python anonymise.py --url <url> --output outputs/result.mp4 --detection-interval 8
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import torch
from tqdm import tqdm

from pipeline.config import PipelineConfig, get_device
from pipeline.detection.ensemble import DetectionEnsemble
from pipeline.extractor import frame_generator, total_frame_count
from pipeline.ingestion import ingest
from pipeline.reconstruction import VideoReconstructor, build_output_path
from pipeline.redaction import Redactor
from pipeline.schemas import VideoMetadata
from pipeline.temporal import TemporalSmoother
from pipeline.tracking.tracker import MultiTracker


@click.command()
@click.option(
    "--url", required=True,
    help="Public video URL (YouTube, direct MP4) or local file path."
)
@click.option(
    "--output", default=None, type=click.Path(),
    help="Output path. Defaults to outputs/anon_<source>.mp4"
)
@click.option(
    "--detection-interval", "detection_interval", default=5, show_default=True,
    help="Run detectors every K frames. Raise K for higher throughput."
)
@click.option(
    "--device", default="auto", show_default=True,
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    help="Compute device. 'auto' selects CUDA > MPS > CPU."
)
@click.option(
    "--face-confidence", "face_confidence", default=0.45, show_default=True,
    help="Face detection confidence threshold [0–1]."
)
@click.option(
    "--logo-confidence", "logo_confidence", default=0.01, show_default=True,
    help="Logo detection confidence threshold [0–1]. Lower = more recall."
)
@click.option(
    "--max-frames", "max_frames", default=None, type=int,
    help="Stop after N frames. For development and quick testing."
)
@click.option(
    "--evaluate", is_flag=True, default=False,
    help="Run evaluation metrics after processing (requires ground truth)."
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False,
    help="Print per-frame detection counts."
)
def main(
    url: str,
    output: str | None,
    detection_interval: int,
    device: str,
    face_confidence: float,
    logo_confidence: float,
    max_frames: int | None,
    evaluate: bool,
    verbose: bool,
) -> None:
    """Detect and redact faces, brand text, and logos from any public video."""

    # ── Config ────────────────────────────────────────────────────────────────
    resolved_device = (
        get_device() if device == "auto" else torch.device(device)
    )
    config = PipelineConfig(
        device=resolved_device,
        detection_interval=detection_interval,
        face_confidence=face_confidence,
        logo_confidence=logo_confidence,
    )

    click.echo("=" * 60)
    click.echo("Video Anonymization Pipeline")
    click.echo(f"  {config.describe()}")
    click.echo("=" * 60)

    # ── Stage 1: Ingestion ────────────────────────────────────────────────────
    click.echo("\n[1/5] Ingesting video...")
    meta = ingest(url, config)

    # ── Stage 2: Initialize pipeline components ───────────────────────────────
    click.echo("\n[2/5] Initializing pipeline...")
    ensemble = DetectionEnsemble(config)
    ensemble.warmup()

    tracker = MultiTracker(config)
    smoother = TemporalSmoother(config)
    redactor = Redactor(config)

    # ── Stage 3: Open video reconstructor ────────────────────────────────────
    output_path = Path(output) if output else build_output_path(meta, config)
    reconstructor = VideoReconstructor(config, meta)
    reconstructor.open(output_path)
    click.echo(f"  Output → {output_path}")

    # ── Stage 4: Main processing loop ────────────────────────────────────────
    click.echo(f"\n[3/5] Processing{f' (max {max_frames} frames)' if max_frames else ''}...")
    n_frames, output_file = _run_pipeline(
        meta, config, ensemble, tracker, smoother, redactor,
        reconstructor, max_frames, verbose,
    )

    # ── Stage 5: Summary ──────────────────────────────────────────────────────
    click.echo(f"\n[4/5] Done — {output_file}")
    ensemble.print_latency_report()

    # ── Optional evaluation ───────────────────────────────────────────────────
    if evaluate:
        click.echo("\n[5/5] Evaluation requested — run evaluate.py separately with annotations.")

    click.echo("=" * 60)


def _run_pipeline(
    meta: VideoMetadata,
    config: PipelineConfig,
    ensemble: DetectionEnsemble,
    tracker: MultiTracker,
    smoother: TemporalSmoother,
    redactor: Redactor,
    reconstructor: VideoReconstructor,
    max_frames: int | None,
    verbose: bool,
) -> tuple[int, Path]:
    """
    Core frame processing loop.

    Data flow per frame:
      extractor → detection gate → tracker update/predict
                → temporal smooth → redact → encode

    Returns (frames_processed, output_path).
    """
    n_estimated = total_frame_count(meta)
    if max_frames:
        n_estimated = min(n_estimated, max_frames)

    frames_processed = 0
    output_path: Path | None = None

    start = time.perf_counter()

    try:
        with tqdm(total=n_estimated or None, unit="frame", dynamic_ncols=True) as pbar:
            for frame in frame_generator(meta, config, max_frames=max_frames):

                # ── Detection gate ────────────────────────────────────────────
                is_det_frame = ensemble.is_detection_frame(frame)
                frame.is_detection_frame = is_det_frame

                if is_det_frame:
                    # Run all three detectors (sparse)
                    frame.detections = ensemble.run(frame)

                    # Update tracker with new detections.
                    # Scene cut resets tracker state before updating —
                    # ordering matters: reset → update (not update → reset).
                    frame.active_tracks = tracker.update(
                        frame.detections,
                        is_scene_cut=frame.is_scene_cut,
                        frame_idx=frame.frame_idx,
                    )
                else:
                    # Non-detection frame: replay last known track positions.
                    # Tracker is not called — no state change, no age increment.
                    frame.active_tracks = tracker.get_active_tracks()

                # ── Temporal smoothing + box expansion ────────────────────────
                # Applied on every frame, detection or not.
                frame.active_tracks = smoother.process(
                    frame.active_tracks,
                    frame_w=frame.width,
                    frame_h=frame.height,
                )

                # ── Redaction ─────────────────────────────────────────────────
                # Modifies frame.image in-place.
                redactor.redact(frame.image, frame.active_tracks)

                # ── Encode ────────────────────────────────────────────────────
                reconstructor.write_frame(frame.image)

                frames_processed += 1

                # Update progress bar
                face_n = sum(1 for t in frame.active_tracks if t.class_name == "face")
                text_n = sum(1 for t in frame.active_tracks if t.class_name == "text")
                logo_n = sum(1 for t in frame.active_tracks if t.class_name == "logo")

                pbar.set_postfix(
                    det="Y" if is_det_frame else "·",
                    cut="!" if frame.is_scene_cut else " ",
                    face=face_n,
                    text=text_n,
                    logo=logo_n,
                    refresh=False,
                )
                pbar.update(1)

    except KeyboardInterrupt:
        click.echo("\n  Interrupted — finalising output with frames so far...")

    finally:
        output_path = reconstructor.close()

    elapsed = time.perf_counter() - start
    fps = frames_processed / elapsed if elapsed > 0 else 0
    click.echo(f"  {frames_processed} frames | {elapsed:.1f}s | {fps:.1f} FPS")

    return frames_processed, output_path


if __name__ == "__main__":
    main()
