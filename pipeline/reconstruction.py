"""
Video reconstruction: encode redacted frames back to MP4 with audio.

Architecture: two-pass FFmpeg.
  Pass 1: pipe raw BGR frames → FFmpeg → temp video-only file (fast H.264 encode)
  Pass 2: FFmpeg mux temp video + source audio → final output (stream copy, no re-encode)

Why two passes: avoids complex synchronisation between frame pipe and audio input.
Pass 2 adds ~0.5s for a typical clip — negligible vs. total pipeline runtime.

Encoder selection:
  macOS:  h264_videotoolbox (Apple VideoToolbox hardware encoder, near-zero CPU cost)
  Linux:  libx264 (standard software encoder, high compatibility)
  Fallback: libx264 always works if VideoToolbox unavailable

Quality: CRF 18 — visually lossless, protects SSIM metric on non-redacted regions.
         Codec-induced SSIM loss at CRF 18 is typically < 0.01.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.schemas import VideoMetadata


class VideoReconstructor:
    """
    Accepts redacted frames one at a time and produces an anonymised MP4.

    Lifecycle:
        rec = VideoReconstructor(config, meta)
        rec.open(output_path)
        for frame in pipeline:
            rec.write_frame(frame.image)
        rec.close()      # flushes encoder + muxes audio
    """

    def __init__(self, config: PipelineConfig, meta: VideoMetadata) -> None:
        self.config = config
        self.meta = meta
        self._proc: subprocess.Popen | None = None
        self._temp_video: Path | None = None
        self._output_path: Path | None = None

    def open(self, output_path: str | Path) -> None:
        """
        Start the FFmpeg encode process and open the frame pipe.
        Must be called before write_frame().
        """
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        # Temp file holds video-only output from pass 1
        self._temp_video = self.config.temp_dir / f"_vidonly_{self._output_path.stem}.mp4"

        encoder = _select_encoder()
        print(f"  Video encoder: {encoder}")

        w, h = self.meta.width, self.meta.height
        fps = self.meta.fps

        cmd = [
            "ffmpeg", "-y",
            # Input: raw BGR frames piped via stdin
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "pipe:0",
            # Output: H.264 MP4
            "-vcodec", encoder,
            "-crf", "18",
            # yuv420p: required for compatibility with QuickTime, VLC, browsers.
            # Without this flag, some encoders use yuv444p which breaks playback.
            "-pix_fmt", "yuv420p",
            str(self._temp_video),
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write one BGR frame to the encoder pipe.
        Frame must match the dimensions in VideoMetadata.
        Raises RuntimeError if the encoder process has died unexpectedly.
        """
        if self._proc is None:
            raise RuntimeError("Call open() before write_frame()")

        if self._proc.poll() is not None:
            stderr = self._proc.stderr.read().decode(errors="replace")
            raise RuntimeError(
                f"FFmpeg encoder died unexpectedly (exit {self._proc.returncode}).\n{stderr}"
            )

        self._proc.stdin.write(frame.tobytes())

    def close(self) -> Path:
        """
        Flush the encoder, close the pipe, and mux audio from the source.
        Returns the path to the final output file.
        """
        if self._proc is None:
            raise RuntimeError("Reconstructor was never opened.")

        # Signal end of input stream — FFmpeg finalises the file
        self._proc.stdin.close()
        returncode = self._proc.wait()

        if returncode != 0:
            stderr = self._proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"FFmpeg encode failed (exit {returncode}):\n{stderr}")

        # Pass 2: mux audio if source has an audio stream
        if self.meta.has_audio:
            print("  Muxing audio...")
            _mux_audio(
                video_path=self._temp_video,
                audio_source_path=Path(self.meta.local_path),
                output_path=self._output_path,
            )
            self._temp_video.unlink(missing_ok=True)
        else:
            # No audio — just move the video-only file to the final path
            shutil.move(str(self._temp_video), str(self._output_path))

        print(f"  Output: {self._output_path}")
        return self._output_path


def build_output_path(meta: VideoMetadata, config: PipelineConfig) -> Path:
    """
    Derive a default output path from the source video filename.
    e.g. .tmp/interview.mp4 → outputs/anon_interview.mp4
    """
    stem = Path(meta.local_path).stem
    return config.output_dir / f"anon_{stem}.mp4"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _select_encoder() -> str:
    """
    Select the best available H.264 encoder on this system.

    h264_videotoolbox: Apple VideoToolbox hardware encoder (macOS only).
    Offloads encode to dedicated hardware — near-zero CPU overhead, fast.

    libx264: Software encoder, always available where FFmpeg is installed.
    High quality, universally compatible output.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        if "h264_videotoolbox" in result.stdout:
            return "h264_videotoolbox"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return "libx264"


def _mux_audio(
    video_path: Path,
    audio_source_path: Path,
    output_path: Path,
) -> None:
    """
    Mux video from video_path with audio from audio_source_path into output_path.

    Both streams are stream-copied (no re-encode) — fast and lossless.
    -shortest: output duration matches the shorter of video or audio.
    This handles the common case where processed video is slightly shorter
    than the source due to frame-level rounding.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_source_path),
        "-c:v", "copy",       # stream copy video — no re-encode, preserves CRF 18 quality
        "-c:a", "copy",       # stream copy audio — no re-encode, preserves original quality
        "-map", "0:v:0",      # video from first input (encoded video)
        "-map", "1:a:0",      # audio from second input (original source)
        "-shortest",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio mux failed (exit {result.returncode}):\n{result.stderr}"
        )
