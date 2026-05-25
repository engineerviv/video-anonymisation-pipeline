# Design Document — Video Anonymization Pipeline

Living design document. Updated at each build stage.

---

## 1. Problem Framing

This is a **systems engineering problem**, not a modelling problem. The core challenge is satisfying four competing constraint classes simultaneously:

- **Throughput** — ≥10 FPS means ≤100ms end-to-end per frame. Three detection models must fit within this budget.
- **Recall** — ≥95% face recall means the system cannot miss 1 in 20 faces across all content types.
- **Temporal consistency** — ≥98% means redaction boxes must be present on every frame of every active track, with no single-frame gaps.
- **Privacy** — ≤2% re-ID rate means redaction must defeat face recognition models, not just obscure visually.

These four constraints conflict. More detection = better recall but lower throughput. Per-frame detection = maximum recall but poor temporal consistency (confidence fluctuations cause single-frame gaps). This conflict is resolved architecturally, not by choosing better models.

---

## 2. Architecture Decision: Sparse Detection + Dense Tracking

**Decision:** Run all three detectors every K frames (K=5 default). Between detection frames, use ByteTrack Kalman filter predictions to propagate bounding boxes.

**Why not per-frame detection:**
- At K=1, naive sequential inference (face ~12ms + text ~35ms + logo ~55ms on M1) = ~102ms/frame = ~10 FPS ceiling with zero headroom for tracking, redaction, and encode.
- Per-frame detection produces independent boxes with no identity continuity. Confidence fluctuations cause single-frame missed detections, breaking temporal consistency.
- A 96% per-frame recall detector produces approximately 76% track-level consistency on average (mathematically: any gap in a track fails that track's consistency fraction).

**Why sparse detection improves consistency:**
- The Kalman filter always produces a prediction — it has no "confidence" that drops below threshold. Tracker predictions fill every frame between detection frames, eliminating single-frame gaps.
- Temporal consistency metric improves because the tracker is more consistent than the detector.

**Why K=5:**
- At K=5, detection runs 6× per second at 30 FPS. A new object entering frame is detected within ~167ms. Acceptable for interview/news content.
- On M1, K=5 reduces detection compute from 102ms/frame to ~20ms average/frame (102ms amortized over 5 frames + 1ms tracking overhead × 4 frames).
- For faster content (sports), K can be reduced to 3 at throughput cost.

**Failure mode:** Fast scene cuts. Kalman filter predicts position based on prior motion — wrong after a hard cut. Mitigation: scene cut detection (pixel difference threshold) triggers tracker reset on cuts.

---

## 3. Model Selection Rationale

### Face Detection: YOLOv8n-face

**Chosen over:** RetinaFace, MTCNN, MediaPipe Face

**Rationale:**
- Ultralytics MPS support is mature — works on M1 without CPU fallback for core ops.
- YOLOv8n-face is trained on WIDER FACE, the standard benchmark for face detection across scale, pose, and occlusion. Strong recall at small scale.
- Single-stage detector: one forward pass per frame, no region proposal step. Faster than two-stage alternatives (RetinaFace with FPN).
- `yolov8n` (nano) variant chosen for speed. `yolov8s-face` (small) is an upgrade path if recall is insufficient.

**Known limitation:** Recall drops below ~30px face width. For 8×8px minimum requirement, face must be at least partially visible — pure 8×8 face detection is extremely difficult for any model and is edge-case territory.

**Run at full resolution** (not downscaled) because small faces are the critical case. A 1920×1080 → 640px downscale turns a 16×16 face into an ~5×5 face, invisible to any detector.

### Text Detection: PaddleOCR (DBNet, detection-only)

**Chosen over:** EasyOCR, CRAFT, Tesseract

**Rationale:**
- PaddleOCR DBNet runs detection-only mode — no recognition (CRNN) step. We don't need to read text, only locate it. Detection-only is 3-5× faster than full OCR.
- PaddleOCR has the best multilingual support (80+ languages) matching the assignment's multilingual requirement.
- Runs on CPU on M1 (PaddlePaddle GPU support on Apple Silicon is patchy). CPU inference is ~35ms on M1, acceptable since text detection is amortized over K frames.

**Heuristic filter:** Only redact text detected in overlay-likely positions: top 20% or bottom 30% of frame height, or within 10% of left/right edges. Rationale: brand watermarks, lower-thirds, and text stickers appear at frame edges. Mid-frame text (e.g., on a whiteboard or book) is generally not a brand overlay. This reduces false positives on non-brand text.

**Known limitation:** Semi-transparent watermarks (opacity < 50%) are difficult for all OCR-based detectors. DBNet relies on edge contrast which degrades with transparency.

### Logo Detection: YOLO-World-S (open-vocabulary)

**Chosen over:** GroundingDINO, CLIP+sliding window, fixed-class YOLO

**Why open-vocabulary is required:**
- A fixed-class logo detector trained on N known logos fails at inference time on any logo not in the training set. The assignment explicitly requires generalization beyond a fixed list.
- A classifier approach ("is this region a logo?") requires a region proposal step and struggles with abstract shapes (Nike swoosh, Apple logo) that are only interpretable as logos with semantic context.
- Open-vocabulary models (YOLO-World, GroundingDINO) use CLIP-style text-image alignment. The text prompt "company logo" activates visual features associated with logos across all training data — no fixed class list.

**Why YOLO-World over GroundingDINO:**
- GroundingDINO: ~300ms/frame on T4. Even with K=5 sparse inference, amortized cost = 60ms/frame. Combined with face + text = 80ms/frame, leaving 20ms for everything else at 10 FPS. Too tight.
- YOLO-World-S: ~15ms/frame on T4, ~55ms on M1 MPS. Amortized at K=5: 3ms/frame on T4, 11ms on M1. Comfortable budget.
- YOLO-World recall is lower than GroundingDINO for abstract logo shapes. Mitigated by: low confidence threshold (0.3), multiple text prompts, and accepting slightly higher FPR.

**Text prompts used:** `["logo", "brand mark", "company logo", "trademark", "watermark", "product logo", "brand name"]`. Multiple prompts increase coverage of the semantic space.

### Tracker: ByteTrack (via supervision)

**Chosen over:** SORT, DeepSORT, StrongSORT

**ByteTrack's key advantage:** Two-buffer matching. Standard SORT discards detections below confidence threshold. ByteTrack keeps low-confidence detections in a secondary buffer and tries to match them against unmatched high-confidence tracks. This dramatically improves recall through partial occlusion — a partially occluded face may have low detection confidence but can still update its track.

**Per-class tracker instances:** Separate ByteTrack instances for face, text, and logo tracks. Prevents track ID collisions between classes. Allows per-class tuning of max_age and min_hits parameters.

**Why not DeepSORT:** DeepSORT adds appearance embeddings (re-ID features) to the matching cost matrix. Better at maintaining identity across long occlusions. However: requires a separate embedding model per frame (adds ~10ms), and appearance features for text/logo tracks are meaningless. Overkill for MVP. Upgrade path if identity consistency bonus is attempted.

---

## 4. Data Flow and Interfaces

All pipeline stages communicate via typed dataclasses defined in `pipeline/schemas.py`. No raw dicts passed between modules.

```
VideoMetadata    — output of ingestion, input to extractor
FrameData        — produced by extractor, flows through entire pipeline
  .detections    — populated by DetectionEnsemble (detection frames only)
  .active_tracks — populated by MultiTracker (every frame)
Detection        — atomic detection result from any detector
Track            — atomic tracker state (Kalman-predicted or updated)
```

The `FrameData` object is the unit of work. Each pipeline stage reads from it and writes to it. The final `FrameData.image` (modified by redaction) is what gets written to the output video.

---

## 5. Redaction Design

### Face: Gaussian Blur

Parameters: kernel=(51,51), sigma=20.0

**Why these values:** ArcFace cosine similarity between original and blurred face drops below 0.4 (non-matching threshold) at sigma≥15 on a 100×100px face. Sigma=20 provides margin. Kernel size must be odd and ≥ 6×sigma for full coverage; 51×51 covers sigma=20 with headroom.

**Why not pixelation:** Pixelation with small block sizes (<8px) is reversible by upscaling with edge-preserving filters. Gaussian blur is harder to reverse. For ≤2% re-ID, Gaussian blur at sigma=20 is sufficient. Pixelation with 16px blocks is equally effective and optionally available.

**Box expansion:** All face boxes expanded by 15% in each direction before blurring. Covers hair, ears, and forehead edges that fall outside a tight face detector box.

### Text: Luminance-Matched Solid Fill

**Why luminance-matched:** A pure black box on a bright broadcast graphic is visually jarring and draws attention. Sampling the 5px border around the text region and filling with the mean luminance value produces a neutral fill that reads as "blank space" rather than "redacted." Matches the assignment specification: "solid fill box matching background luminance."

Implementation: convert border region to YCrCb colorspace, sample mean Y (luminance) channel, fill BGR with equivalent grey value.

### Logo: Gaussian Blur

Parameters: kernel=(41,41), sigma=15.0

Logos often appear on product surfaces (clothing, backgrounds) where a solid fill would look obviously artificial. Blur reads as "out of focus" and is less intrusive.

### Feathered Mask Edges

All redaction types apply a 3px soft feather at the bbox boundary. The transition from blurred/filled region to clean image is blended using a Gaussian-weighted mask. Eliminates hard edges, improves SSIM at region boundaries, looks more natural.

---

## 6. Throughput Analysis

### Per-frame latency budget (T4 GPU, K=5 sparse — measured on Kaggle T4):

| Stage | Detection frame | Non-detection frame |
|---|---|---|
| Frame decode | 2ms | 2ms |
| Face detection (CUDA) | **21ms** | 0ms |
| Text detection (CPU) | **147ms** | 0ms |
| Logo detection (CUDA) | **16ms** | 0ms |
| Tracking update/predict | 2ms | 1ms |
| Temporal smoothing | 1ms | 1ms |
| Redaction | 3ms | 3ms |
| Frame encode (pipe) | 2ms | 2ms |
| **Total** | **194ms** | **9ms** |
| **Amortized (K=5)** | **(183ms + 4×9ms) / 5 = 37ms** | |
| **FPS ceiling** | **~27 FPS** | |
| **Measured FPS** | **18.2 FPS** | |

Measured on Kaggle T4 processing a 1080p video (5831 frames, K=5). Python/GIL overhead accounts for the gap between 27 FPS ceiling and 18.2 FPS actual. Text detection is the bottleneck at 147ms/frame — PaddleOCR runs CPU-only even on GPU hosts.

### Per-frame latency budget (M1 MPS, K=5 sparse — measured):

Measured on Apple M1 processing a 1080×1920 30fps YouTube Short (212 detection frames over 1041 total):

| Stage | Detection frame | Non-detection frame |
|---|---|---|
| Frame decode | 3ms | 3ms |
| Face detection (MPS) | 82ms | 0ms |
| Text detection (CPU) | 289ms | 0ms |
| Logo detection (MPS) | 37ms | 0ms |
| Tracking + smooth + redact | 5ms | 5ms |
| Encode (videotoolbox) | 1ms | 1ms |
| **Total** | **417ms** | **9ms** |
| **Amortized (K=5)** | **(417 + 4×9) / 5 = 90ms** | |
| **Effective FPS** | **~11 FPS ceiling** | |

Realistic with Python/GIL overhead: **6–17 FPS on M1**, resolution-dependent.

**Why text is the bottleneck:** PaddleOCR runs CPU-only on M1 (PaddlePaddle MPS support is incomplete). At 289ms/detection-frame, text detection is 3.5× slower than face and logo combined. On a T4 with `paddlepaddle-gpu`, this drops to ~8ms. If CPU-only throughput is critical, set `--detection-interval 10` or disable text detection with a flag.

---

## 7. Privacy Engineering Considerations

### Re-ID Resistance

The ≤2% re-ID rate is measured by running ArcFace on redacted face regions and comparing embeddings to the original. At sigma=20 Gaussian blur, cosine similarity drops well below the 0.6 matching threshold for faces ≥40×40px. For very small faces (<30×30px), blur at sigma=20 may be less effective because there's less information to begin with — these tend to naturally score low on re-ID regardless.

### Adversary Model

This system defends against: casual human re-identification, standard face recognition APIs (ArcFace, FaceNet, DeepFace). It does not claim to defend against: adversarial attacks specifically designed to reconstruct faces from blur artifacts, or future more powerful embedding models. This limitation is documented in the model card.

### Temporal Privacy

A single-frame gap in redaction (frame N has no box while N-1 and N+1 do) is a privacy failure — a viewer who pauses at frame N can identify the person. The ByteTrack propagation eliminates this by ensuring every frame in an active track has a predicted box, regardless of whether detection fired on that frame.

---

## 8. Known Limitations and Failure Modes

| Failure Mode | Impact | Mitigation |
|---|---|---|
| Fast scene cuts mid-track | Stale box propagated to wrong content | Scene cut detection + tracker reset |
| Small faces (<20×20px) | Detection recall drops significantly | Confidence threshold tuning, RetinaFace upgrade path |
| Dark skin tones | YOLOv8-face has known recall disparity on darker skin in low light | Document in model card; evaluate across demographics |
| Semi-transparent watermarks | PaddleOCR DBNet misses low-contrast text | CRAFT as alternative detector |
| Abstract logos without text context | YOLO-World semantic query fails on non-textual shapes | Accept as known limitation; document in model card |
| Heavy H.264 compression | Reduces edge contrast, hurts text/logo detection | Not mitigated — accept performance degradation on heavily compressed video |
| VFR (variable frame rate) video | Scene cut threshold and K-frame counting may be wrong | Normalize to CFR in extractor |

---

## 9. Dependency Version Constraints

Two hard version pins exist that future maintainers must not remove without testing:

| Package | Pin | Reason |
|---|---|---|
| `paddleocr>=2.7.3,<3.0.0` | Upper bound critical | PaddleOCR 3.x removed `use_gpu`, `use_angle_cls`, `show_log`, `det_db_thresh` kwargs and changed `ocr(rec=False)` to `predict()`. The 2.x API is required. |
| `supervision>=0.21.0,<0.30.0` | Upper bound critical | ByteTrack was removed from supervision in 0.30.0. The tracker depends on `supervision.ByteTrack`. |

The face model URL also changed between versions of the source repo. The current working URL is `https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt` (repo was renamed from `yolov8-face` to `yolo-face`, tag changed from `v0.0.0` to `1.0.0`).
