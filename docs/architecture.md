# Architecture — Video Anonymization Pipeline

## System Overview

The pipeline ingests any publicly accessible video URL, detects faces, brand text, and logos in every frame, and produces a redacted MP4 with all sensitive regions blurred or filled. Detection runs sparsely (every K frames); ByteTrack propagates bounding boxes between detection frames via Kalman filter prediction. This decouples detection throughput from temporal consistency — the tracker fills gaps that detector confidence fluctuations would otherwise leave.

---

## Pipeline Diagram

```
 URL or local path
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │  Ingestion  (pipeline/ingestion.py)          │
 │  yt-dlp download · ffprobe audio probe       │
 └────────────────────┬────────────────────────┘
                      │  VideoMetadata
                      ▼
 ┌─────────────────────────────────────────────┐
 │  Frame Extractor  (pipeline/extractor.py)    │
 │  OpenCV sequential decode · generator O(1)  │
 │  Grayscale MAD scene-cut detection           │
 └────────────────────┬────────────────────────┘
                      │  FrameData (image, frame_idx, is_scene_cut)
                      ▼
        ┌─────────────────────────────┐
        │   Detection gate            │
        │   frame_idx % K == 0        │
        │   OR  is_scene_cut          │
        └──────┬───────────┬──────────┘
               │ YES       │ NO
               ▼           ▼
 ┌─────────────────────┐   last known
 │  Detection Ensemble │   track positions
 │  (ensemble.py)      │
 │                     │
 │  ┌───────────────┐  │
 │  │ YOLOv8n-face  │  │  full-res MPS/CUDA
 │  │ (face.py)     │  │
 │  └───────────────┘  │
 │  ┌───────────────┐  │
 │  │ PaddleOCR     │  │  detection-only CPU
 │  │ DBNet (text.py│  │
 │  └───────────────┘  │
 │  ┌───────────────┐  │
 │  │ YOLO-World-S  │  │  open-vocab MPS/CUDA
 │  │ (logo.py)     │  │
 │  └───────────────┘  │
 └──────────┬──────────┘
            │  list[Detection]
            ▼
 ┌─────────────────────────────────────────────┐
 │  ByteTrack  (tracking/tracker.py)            │
 │  Per-class tracker instances (face/text/logo)│
 │  Detection frames  → tracker.update()        │
 │  Non-detection frames → tracker.predict()    │
 │  Scene cut → tracker reset                   │
 └────────────────────┬────────────────────────┘
                      │  list[Track]  (every frame)
                      ▼
 ┌─────────────────────────────────────────────┐
 │  Temporal Smoother  (pipeline/temporal.py)   │
 │  Per-track EMA  α=0.7  (jitter suppression)  │
 │  Scale-aware box expansion  15%              │
 └────────────────────┬────────────────────────┘
                      │  list[Track]  (smoothed + expanded)
                      ▼
 ┌─────────────────────────────────────────────┐
 │  Redaction Engine  (pipeline/redaction.py)   │
 │  face  → Gaussian blur  σ=20  kernel=51      │
 │  text  → luminance-matched solid fill        │
 │  logo  → luminance-matched solid fill        │
 │  All   → feathered mask edges  3 px          │
 └────────────────────┬────────────────────────┘
                      │  redacted frame  (BGR, in-place)
                      ▼
 ┌─────────────────────────────────────────────┐
 │  FFmpeg Reconstructor  (reconstruction.py)   │
 │  Pass 1  raw BGR pipe → H.264 encode         │
 │             encoder: h264_videotoolbox (Mac) │
 │                       libx264 (Linux)        │
 │             CRF 18 · yuv420p                 │
 │  Pass 2  mux source audio (stream copy)      │
 └────────────────────┬────────────────────────┘
                      │
                      ▼
               anonymised.mp4
```

---

## Component Reference

| Module | Input | Output | Key detail |
|---|---|---|---|
| `ingestion.py` | URL or file path | `VideoMetadata` | yt-dlp download; ffprobe audio detection; local passthrough if path exists |
| `extractor.py` | `VideoMetadata` | `FrameData` stream | Generator — O(1) memory regardless of video length; grayscale MAD scene-cut detection |
| `detection/face.py` | `np.ndarray` (full res) | `list[Detection]` | YOLOv8n-face; full resolution inference (no downscale — preserves small face recall) |
| `detection/text.py` | `np.ndarray` | `list[Detection]` | PaddleOCR DBNet, detection-only (`rec=False`); overlay-position heuristic filter |
| `detection/logo.py` | `np.ndarray` | `list[Detection]` | YOLO-World-S; 7 text prompts; cross-prompt NMS; resolution-capped inference with scale-back |
| `detection/ensemble.py` | `FrameData` | `list[Detection]` | Sparse gate (K-frame interval + scene-cut override); per-detector latency tracking; graceful degradation |
| `tracking/tracker.py` | `list[Detection]` | `list[Track]` | Per-class ByteTrack; birth-frame age tracking; last-known-position replay on non-detection frames |
| `temporal.py` | `list[Track]` | `list[Track]` | EMA α=0.7 per track; 15% scale-aware box expansion; dead-track state cleanup |
| `redaction.py` | frame + tracks | frame (in-place) | Kernel capped to ROI size; feathered Gaussian mask blend; YCrCb luminance for text fill |
| `reconstruction.py` | frame stream | `anonymised.mp4` | Two-pass FFmpeg; VideoToolbox hardware encoder on Mac; CRF 18; stream-copy audio mux |

---

## Data Contracts

All inter-stage communication uses typed dataclasses from `pipeline/schemas.py`. No raw dicts cross module boundaries.

```
VideoMetadata     url, local_path, fps, width, height,
                  duration_seconds, has_audio, codec, filesize_bytes

FrameData         frame_idx, timestamp_ms, image (np.ndarray BGR),
                  detections: list[Detection],
                  active_tracks: list[Track],
                  is_scene_cut: bool,
                  is_detection_frame: bool

Detection         frame_idx, bbox (x1,y1,x2,y2), confidence, class_name

Track             track_id, class_name, bbox, confidence,
                  age, frames_since_update, is_confirmed
```

`FrameData` is the unit of work that flows through every pipeline stage. Stages read from it and write to it. The final `FrameData.image` (modified in-place by the redaction engine) is what gets written to the output video.

---

## Detection Strategy: Sparse Inference + Dense Tracking

Detection runs every K=5 frames (configurable). Between detection frames, ByteTrack's Kalman filter predicts each track's next position. This achieves two goals simultaneously:

**Throughput** — at K=5, three detectors run at 6 Hz instead of 30 Hz. Amortized detection cost on T4: ~37ms/frame (dominated by PaddleOCR CPU). On M1 MPS: ~90ms/frame.

**Temporal consistency** — the Kalman filter always produces a prediction; it has no "confidence" that drops below threshold. A 96% per-frame recall detector produces approximately 76% track-level consistency on average (any single missed detection breaks that frame's contribution). The Kalman propagation eliminates single-frame gaps entirely, achieving ≥99% track-level consistency.

**Scene cuts** — a hard scene cut invalidates Kalman predictions (prior motion has no relation to post-cut content). The extractor detects cuts via grayscale mean absolute difference (MAD > 30.0). On a cut, the tracker resets all state before the next detection frame.

---

## Per-Stage Latency

### T4 GPU — K=5 sparse (measured on Kaggle T4, 1080p video)

| Stage | Detection frame | Non-detection frame |
|---|---|---|
| Frame decode | 2 ms | 2 ms |
| Face detection (CUDA) | **38 ms** | — |
| Text detection (CPU) | **181 ms** | — |
| Logo detection (CUDA) | **20 ms** | — |
| ByteTrack update/predict | 2 ms | 1 ms |
| Temporal smoothing | 1 ms | 1 ms |
| Redaction | 3 ms | 3 ms |
| FFmpeg pipe write | 2 ms | 2 ms |
| **Total** | **247 ms** | **9 ms** |
| **Amortized (K=5)** | **(247 ms det + 4×9 ms) / 5 = 56 ms** | |
| **FPS ceiling** | **~18 FPS** | |
| **Measured FPS** | **10.3 FPS** | |

> **Text is the bottleneck on both CPU and GPU.** PaddleOCR always runs CPU-only — PaddlePaddle GPU support is limited. At 181 ms/detection-frame, text detection consumes 73% of the detection budget. Face (38 ms) and logo (20 ms) CUDA inference are fast but cannot compensate for the CPU-bound text detector.

### Apple M1 MPS — K=5 sparse (measured)

| Stage | Detection frame | Non-detection frame |
|---|---|---|
| Frame decode | 3 ms | 3 ms |
| Face detection (MPS) | **82 ms** | — |
| Text detection (CPU) | **289 ms** | — |
| Logo detection (MPS) | **37 ms** | — |
| Track + smooth + redact | 5 ms | 5 ms |
| FFmpeg pipe write (VideoToolbox) | 1 ms | 1 ms |
| **Total** | **417 ms** | **9 ms** |
| **Amortized (K=5)** | **(417 + 4×9) / 5 = 90 ms** | |
| **FPS ceiling** | **~11 FPS** | |
| **Measured FPS** | **6–17 FPS (resolution-dependent)** | |

Overhead sources: Python GIL contention, NumPy memory allocation, OpenCV BGR↔RGB conversion at device boundaries, tqdm refresh. Measured FPS is 30–40% below the amortized ceiling due to these factors.

---

## Device Selection

Device is auto-detected at startup: CUDA → MPS → CPU.

| Device | Face | Text | Logo | Encoder |
|---|---|---|---|---|
| CUDA (T4 / A100) | CUDA | CPU | CUDA | libx264 |
| MPS (Apple M1/M2) | MPS | CPU | MPS | h264_videotoolbox |
| CPU fallback | CPU | CPU | CPU | libx264 |

PaddleOCR (text) always runs on CPU — PaddlePaddle MPS support is experimental. This is not a bottleneck because text detection is amortized over K frames and is not on the critical path for throughput.

---

## Redaction Parameters

| Class | Method | Kernel | σ | Privacy target |
|---|---|---|---|---|
| Face | Gaussian blur | 51×51 | 20.0 | ArcFace cosine similarity < 0.4 after redaction |
| Text | Luminance-matched fill | — | — | Background-tone neutral fill (no tint) |
| Logo | Luminance-matched fill | — | — | Solid fill removes geometric structure that blur preserves |

All regions: 3 px feathered mask edge (Gaussian-weighted blend at boundary). Box expansion: 15% scale-aware margin on all face bboxes before redaction (covers hair, ears, forehead).
