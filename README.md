# Video Anonymisation Pipeline

Automatically detects and redacts **faces**, **brand text**, and **logos** from any publicly accessible internet video — without prior knowledge of content, language, or domain.

---

## Assignment Scope

| Target | Method | Threshold |
|---|---|---|
| Human faces (including partial, side-profile, small-scale ≥8×8px) | Gaussian blur σ≥20 | ≥95% recall, ≥90% precision |
| Text overlays, watermarks, lower-thirds (multilingual) | Luminance-matched solid fill | ≥90% recall |
| Company logos, brand marks (open-vocabulary) | Gaussian blur or fill | ≥90% recall |

### Performance Targets

| Metric | Pass | Distinction |
|---|---|---|
| Face Recall | ≥95% | ≥97% |
| Face Precision | ≥90% | ≥90% |
| Text/Logo Recall | ≥90% | ≥93% |
| Re-ID Rate | ≤2% | ≤1% |
| Temporal Consistency | ≥98% | ≥98% |
| Throughput (T4) | ≥10 FPS | ≥20 FPS |
| Visual Quality (SSIM) | ≥0.85 | ≥0.85 |
| False Positive Rate | ≤5% | ≤5% |

### Measured Results (M1 Mac, 10 test videos)

| Metric | Value | Status |
|---|---|---|
| Face Recall | 63.4% | — (synthetic + dashcam clips structurally 0%) |
| Face Precision | 99.2% | ✅ |
| Re-ID Rate | 0.0% | ✅ |
| Temporal Consistency | 94.1% | — |
| SSIM (non-redacted) | 99.9% | ✅ |
| False Positive Rate | 0.0% | ✅ |
| Throughput (M1) | 6–17 FPS | — (GPU target: 25–40 FPS) |

> Face recall on real frontal-face videos specifically: 60–82%. Aggregate is pulled down by synthetic clips (cartoon shapes not recognised by a real-photo-trained model) and a dashcam clip with no frontal faces. See `outputs/eval/benchmark_report.pdf` for full failure analysis.

---

## Architecture Overview

**Sparse Detection + ByteTrack Propagation**

```
URL → [Ingestion] → [Frame Extractor] → [Detection Ensemble (every K frames)]
                                      → [ByteTrack per class]
                                      → [Temporal Smoother]
                                      → [Redaction Engine]
                                      → [FFmpeg Reconstruction]
                                      → anonymised.mp4
```

Detection runs every K frames (default K=5). ByteTrack propagates bounding boxes between detection frames via Kalman filter prediction. This achieves temporal consistency and throughput simultaneously.

See `docs/design.md` for full architectural rationale and component design decisions.

---

## Hardware

| Environment | Device | Expected Throughput |
|---|---|---|
| Apple M1 16GB (development) | MPS / CPU | 6–17 FPS (resolution-dependent) |
| Kaggle T4 (benchmarking) | CUDA | 25–40 FPS |
| A100 (production target) | CUDA | 50–80 FPS |

Throughput varies significantly with resolution: 640×360 → ~17 FPS, 1080×1920 portrait → ~7 FPS on M1.

Device is auto-detected at runtime: CUDA → MPS → CPU.

---

## Model Stack

| Component | Model | Backend | Notes |
|---|---|---|---|
| Face detection | YOLOv8n-face | MPS / CUDA | WIDER FACE trained, strong small-face recall |
| Text detection | PaddleOCR (DBNet) | CPU | Detection-only mode, no recognition needed |
| Logo detection | YOLO-World-S | MPS / CUDA | Open-vocabulary via CLIP text-image alignment |
| Tracking | ByteTrack (supervision) | CPU | Per-class tracker instances |
| Re-ID evaluation | InsightFace ArcFace | CPU | Evaluation only, not in main pipeline |
| Video encode | FFmpeg h264 | HW accel on M1 | h264_videotoolbox on Mac, libx264 on Linux |

---

## Project Structure

```
video-anonymisation-pipeline/
├── anonymise.py              # CLI entrypoint: python anonymise.py --url <url>
├── evaluate.py               # Evaluation CLI
├── requirements.txt
├── environment.yml
├── Dockerfile
├── setup_kaggle.sh           # One-shot Kaggle environment setup
│
├── pipeline/
│   ├── config.py             # PipelineConfig + device abstraction
│   ├── schemas.py            # Detection, Track, FrameData dataclasses
│   ├── ingestion.py          # yt-dlp video download + metadata
│   ├── extractor.py          # Frame decode, scene cut detection
│   ├── temporal.py           # EMA smoothing, box expansion, gap fill
│   ├── redaction.py          # Blur/fill per detection class
│   ├── reconstruction.py     # FFmpeg encode + audio mux
│   ├── detection/
│   │   ├── base.py           # BaseDetector abstract class
│   │   ├── face.py           # YOLOv8n-face
│   │   ├── text.py           # PaddleOCR detection-only
│   │   ├── logo.py           # YOLO-World open-vocabulary
│   │   └── ensemble.py       # Multi-detector orchestrator + sparse gate
│   └── tracking/
│       └── tracker.py        # ByteTrack per-class tracker
│
├── evaluation/
│   ├── metrics.py            # Recall, precision, F1, SSIM, FPS, consistency
│   └── reid_test.py          # ArcFace re-ID resistance measurement
│
├── data/
│   ├── create_test_data.py   # Generates 5 synthetic test videos with exact GT
│   ├── auto_annotate.py      # Proxy GT annotation for real videos
│   ├── make_report.py        # Generates benchmark report PDF
│   ├── test_videos/          # Test clips (gitignored — large files)
│   └── annotations/          # Ground truth JSON annotations (committed)
│
├── outputs/                  # Processed video outputs (gitignored)
├── models/                   # Downloaded model weights (gitignored)
└── docs/
    ├── design.md             # Architecture decisions, tradeoffs, component design
    ├── architecture.md       # Pipeline diagram + latency breakdown (deliverable)
    └── model_card.md         # Biases, failure modes, ethics (deliverable)
```

---

## Setup

### Local (Apple M1)

```bash
conda env create -f environment.yml
conda activate video-anon
```

### Kaggle (T4 GPU)

1. Create a new Kaggle notebook and set **Settings → Accelerator → GPU T4**
2. Run these cells in order:

```python
# Cell 1 — Clone repo
!git clone https://github.com/engineerviv/video-anonymisation-pipeline.git
%cd video-anonymisation-pipeline

# Cell 2 — Install dependencies (3–5 minutes)
!bash setup_kaggle.sh

# Cell 3 — Verify GPU
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Cell 4 — Run the pipeline
!python anonymise.py --url "https://www.youtube.com/watch?v=<id>"
```

> The FPS shown in the tqdm progress bar is the GPU benchmark number. Expect 25–40 FPS on T4 at 720p.

### Docker (Linux/CUDA — for reproducible benchmarking)

```bash
docker build -t video-anon .

# Anonymize a video
docker run --gpus all -v $(pwd)/outputs:/app/outputs \
  video-anon --url <video_url>

# Run full evaluation
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  --entrypoint python video-anon \
  evaluate.py --video-dir data/test_videos/ --annotation-dir data/annotations/
```

> Models are not baked into the image — they download to `/app/models` on first run. Mount a host path (`-v /host/models:/app/models`) to persist them across runs.

---

## Usage

```bash
# Anonymize any public video URL
python anonymise.py --url "https://www.youtube.com/watch?v=<id>"

# With options
python anonymise.py \
  --url <url> \
  --output outputs/result.mp4 \
  --detection-interval 5 \
  --device auto

# Run evaluation on test set (all videos, with annotations)
python evaluate.py \
  --video-dir data/test_videos/ \
  --annotation-dir data/annotations/ \
  --max-frames 300 \
  --no-reid

# Evaluate a single video
python evaluate.py \
  --video data/test_videos/clip.mp4 \
  --annotation data/annotations/clip.json

# Regenerate benchmark report PDF
python data/make_report.py
```

---

## Evaluation Methodology

Ground truth annotations are generated by `data/auto_annotate.py`, which runs all three detectors at reduced confidence thresholds (face=0.25, text=0.30, logo=0.20) on every frame. These are **proxy annotations** — not manual labels — and are biased toward what the detectors can find. They are honest lower bounds, not perfect ground truth.

Detection matching uses **PASCAL VOC IoU ≥ 0.50**, micro-averaged across all test clips. To ensure fair comparison, always run `evaluate.py` with `--max-frames 300` when using these annotations (both are aligned to the first 300 frames).

---

## Key Version Pins

Two dependencies have hard upper-bound pins that must not be removed:

| Package | Pin | Reason |
|---|---|---|
| `paddleocr<3.0.0` | Critical | PaddleOCR 3.x removed `use_gpu`, `ocr(rec=False)`, and other kwargs the text detector uses |
| `supervision<0.30.0` | Critical | ByteTrack was removed from supervision 0.30 — tracker silently breaks |

---

## Deliverables

| # | Deliverable | Location |
|---|---|---|
| 1 | Pipeline code | `anonymise.py` + `pipeline/` |
| 2 | Evaluation script | `evaluate.py` + `evaluation/` |
| 3 | Benchmark report PDF | `outputs/eval/benchmark_report.pdf` |
| 4 | Architecture diagram | `docs/architecture.md` |
| 5 | Model card | `docs/model_card.md` |
| 6 | Reproducible environment | `Dockerfile`, `environment.yml`, `setup_kaggle.sh` |

---

## Bonus Challenges

| Challenge | Status |
|---|---|
| ONNX INT8 edge deployment (≥5 FPS CPU-only) | Pending |
| Consistent identity tokens (same avatar per person) | Pending |
