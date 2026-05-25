"""
Video ingestion: download any public URL via yt-dlp, extract metadata.

Design decisions:
- Local file paths are passed through without download (dev iteration speed).
- Format selector prefers MP4+AAC to avoid re-encoding during audio mux.
- Audio presence is checked via ffprobe — OpenCV cannot reliably detect audio streams.
- Downloaded files land in config.temp_dir (.tmp/) which is gitignored.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import cv2
import yt_dlp

from pipeline.config import PipelineConfig
from pipeline.schemas import VideoMetadata


def ingest(url: str, config: PipelineConfig) -> VideoMetadata:
    """
    Download a video from any public URL and return its metadata.

    If `url` is a path to an existing local file, skip download entirely.
    Supports YouTube, direct MP4/WebM links, and any yt-dlp supported platform.

    Raises:
        RuntimeError: if download fails or output file is not found after download.
    """
    local_path = Path(url)
    if local_path.exists() and local_path.is_file():
        print(f"  Local file detected, skipping download: {url}")
        return _extract_metadata(str(local_path), url)

    print(f"  Downloading: {url}")
    downloaded_path = _download(url, config)
    print(f"  Saved to: {downloaded_path}")

    return _extract_metadata(downloaded_path, url)


# ── Download ──────────────────────────────────────────────────────────────────

def _download(url: str, config: PipelineConfig) -> str:
    """
    Download video using yt-dlp. Returns absolute path to downloaded file.

    Format selector explained:
      bestvideo[ext=mp4]+bestaudio[ext=m4a]  →  separate streams, merged by FFmpeg
      /best[ext=mp4]                          →  single-file MP4 fallback
      /best                                   →  any format last resort

    We prefer MP4+M4A because FFmpeg can mux them without re-encoding the video
    stream (stream copy), preserving quality and maximizing speed.
    """
    output_template = str(config.temp_dir / "%(title).80s.%(ext)s")
    downloaded_path: Optional[str] = None

    def _on_progress(d: dict) -> None:
        nonlocal downloaded_path
        if d["status"] == "finished":
            downloaded_path = d["filename"]
        elif d["status"] == "downloading":
            pct = d.get("_percent_str", "?%").strip()
            speed = d.get("_speed_str", "?/s").strip()
            print(f"\r  Downloading... {pct} at {speed}    ", end="", flush=True)

    ydl_opts: dict = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "noplaylist": True,       # never pull full playlists, only the given video
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
        "progress_hooks": [_on_progress],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            print()  # newline after progress line

            # Primary: path reported by progress hook
            if downloaded_path and Path(downloaded_path).exists():
                return str(Path(downloaded_path).resolve())

            # Fallback: reconstruct from info dict (yt-dlp may rename after merge)
            raw_path = ydl.prepare_filename(info)
            for ext in ("mp4", "mkv", "webm", "m4v"):
                candidate = Path(raw_path).with_suffix(f".{ext}")
                if candidate.exists():
                    return str(candidate.resolve())

    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(
            f"yt-dlp failed to download '{url}'.\n"
            f"Common causes: private video, geo-restriction, unsupported URL.\n"
            f"Original error: {exc}"
        ) from exc

    raise RuntimeError(
        f"Download reported success but output file not found. "
        f"Last known path: {downloaded_path}"
    )


# ── Metadata extraction ───────────────────────────────────────────────────────

def _extract_metadata(video_path: str, url: str) -> VideoMetadata:
    """
    Read video properties from a local file.
    Uses OpenCV for frame-level properties; ffprobe for audio stream detection.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video file: {video_path}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration: float = frame_count / fps if fps > 0 else 0.0
    codec: str = _fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
    cap.release()

    has_audio: bool = _check_has_audio(video_path)
    filesize: int = Path(video_path).stat().st_size

    meta = VideoMetadata(
        url=url,
        local_path=video_path,
        fps=fps,
        width=width,
        height=height,
        duration_seconds=duration,
        has_audio=has_audio,
        codec=codec,
        filesize_bytes=filesize,
    )

    print(
        f"  Metadata: {width}×{height} @ {fps:.2f}fps | "
        f"{duration:.1f}s | codec={codec} | audio={has_audio} | "
        f"{filesize / 1_000_000:.1f}MB"
    )
    return meta


def _fourcc_to_str(fourcc_int: int) -> str:
    """Convert OpenCV FOURCC integer to 4-char codec string (e.g. 'avc1')."""
    chars = [chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]
    result = "".join(c for c in chars if c.isprintable() and c != "\x00")
    return result or "unknown"


def _check_has_audio(video_path: str) -> bool:
    """
    Use ffprobe to detect whether the video file contains an audio stream.
    OpenCV does not expose audio stream information — ffprobe is the right tool.
    Returns False if ffprobe is not installed or times out.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "audio" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # ffprobe not found or timed out — assume no audio, continue safely
        return False
