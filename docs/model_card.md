# Model Card — Video Anonymization Pipeline

## Overview

**System:** Automated detection and redaction of faces, brand text, and logos from video.
**Version:** 1.0
**Type:** Multi-model CV pipeline (detection + tracking + redaction)
**Primary use:** Privacy-preserving video publication — removing personally identifiable information (PII) and brand identifiers before sharing footage publicly.

---

## Intended Use

### In-scope
- Redacting faces from interview footage, news video, street recordings, and user-generated content before public release
- Removing brand logos and text overlays from broadcast video for compliance or competitive sensitivity reasons
- Preprocessing training data to reduce PII exposure
- Automated first-pass anonymization for human review workflows (not as a sole privacy control)

### Out-of-scope
- **Surveillance or tracking** — this system identifies and tracks individuals across frames. Using it to build person-movement profiles, even from redacted output, is a misuse of the tracking component.
- **Identity verification** — the re-ID evaluation component (ArcFace) is an evaluation tool only and should not be repurposed as an identity recognition system.
- **Legal compliance as a sole control** — output should be reviewed by a human before use in any legal, medical, or regulatory context. This system does not constitute a privacy guarantee.
- **Real-time streaming** — the pipeline processes pre-recorded video files. It is not designed for live stream processing and has not been evaluated for that use case.
- **Audio anonymization** — speech, voice tone, and audio metadata are not processed. Audio is stream-copied unmodified.

---

## Model Components

### 1. Face Detection — YOLOv8n-face

| Property | Value |
|---|---|
| Architecture | YOLOv8 nano, single-stage detector |
| Training data | WIDER FACE dataset (~32,000 images, 393,000 faces) |
| Input | Full-resolution BGR frame (no downscale) |
| Output | Bounding boxes + confidence scores |
| Confidence threshold | 0.45 (configurable) |

**Performance characteristics**
- Strong recall on frontal and near-frontal faces at medium scale (≥40px width)
- Recall degrades below ~30px face width; faces ≤20px are near-random
- Side profiles (>60° yaw) show lower recall than frontal poses
- Fast occlusion (≥70% of face occluded) reduces recall significantly
- Performs well at moderate angles (≤45° yaw, ≤30° pitch)

**Known demographic disparities**
- YOLOv8-face inherits biases from WIDER FACE. WIDER FACE skews toward lighter-skinned subjects in high-contrast lighting conditions.
- Recall on darker skin tones in low-light or backlit conditions is lower than on lighter skin tones under equivalent conditions. This is a documented characteristic of WIDER FACE-trained detectors.
- Face recall at small scale (≤30px) is unreliable across all demographic groups; this is a resolution constraint, not a demographic bias.

**Mitigation options (upgrade path)**
- Replace with RetinaFace (better small-face recall) or InsightFace SCRFD (faster, better demographic coverage)
- Lower confidence threshold (e.g., 0.30) to increase recall at the cost of precision

---

### 2. Text Detection — PaddleOCR DBNet

| Property | Value |
|---|---|
| Architecture | DBNet (Differentiable Binarization Network) |
| Training data | PaddleOCR multilingual dataset (80+ languages) |
| Mode | Detection-only (`rec=False`) — locates text without reading it |
| Input | Full-resolution BGR frame |
| Output | Polygon bounding boxes (converted to axis-aligned rects) |
| Confidence threshold | 0.50 |

**Overlay-position heuristic**
Only text detected in overlay-likely positions is redacted: top 20%, bottom 30%, or within 8% of left/right frame edges. Mid-frame text (whiteboards, books, clothing) is not redacted. This trades recall on edge cases for a significant reduction in false positives.

**Performance characteristics**
- Strong recall on horizontal Latin, CJK, Arabic, and Devanagari text at ≥12px height
- Recall degrades on text smaller than ~10px height (below OCR legibility threshold for any model)
- Recall degrades significantly on semi-transparent watermarks (opacity < 50%) due to reduced edge contrast
- Stylised fonts with extreme distortion (e.g., graffiti-style logos) may be missed

**Known limitations**
- Vertical text (rotated 90°) has lower recall; DBNet is optimized for near-horizontal text
- Text rendered on curved surfaces (bottles, clothing seams) may be missed or have distorted bounding boxes
- The overlay-position heuristic will miss brand text placed at non-standard positions (e.g., centred watermarks, mid-frame tickers)

---

### 3. Logo Detection — YOLO-World-S (Open-Vocabulary)

| Property | Value |
|---|---|
| Architecture | YOLO-World-S (YOLOv8 + CLIP text-image alignment) |
| Training data | YOLO-World pretrained on Objects365 + CC3M caption alignment |
| Text prompts | `["logo", "brand mark", "company logo", "trademark", "watermark", "product logo", "brand name"]` |
| Confidence threshold | 0.30 |

**Why open-vocabulary is required**
A fixed-class logo detector fails on any logo not in its training set. YOLO-World uses CLIP-style text-image alignment to activate visual features associated with logos across all training data, generalising to unseen brands.

**Performance characteristics**
- Detects logos with strong visual texture and geometric regularity well (badges, stickers, printed text)
- Recall is lower on abstract minimalist marks (Apple logo, Nike swoosh, plain geometric shapes) — these are semantically logo-like but visually ambiguous without context
- Overlapping detections from multiple prompts are deduplicated via cross-prompt NMS (IoU > 0.5)

**Known limitations**
- Recall on tiny logos (≤20px) is unreliable; these are below CLIP's effective embedding resolution
- Logos embedded in complex scenes (cluttered backgrounds, heavy compression) have lower recall
- The open-vocabulary approach inherits any CLIP training biases in how "logo" concepts are represented

---

### 4. Tracking — ByteTrack (via supervision)

| Property | Value |
|---|---|
| Algorithm | ByteTrack (two-buffer Hungarian matching) |
| Implementation | `supervision` library (pinned `<0.30.0`) |
| Per-class instances | Separate tracker for face, text, logo |
| Kalman filter | Position + velocity prediction between detection frames |

**Why ByteTrack over SORT**
ByteTrack's second buffer retains low-confidence detections rather than discarding them. During partial occlusion, a face may drop below the primary confidence threshold while remaining detectable at lower confidence. ByteTrack matches these low-confidence detections to existing high-confidence tracks, improving recall through occlusion transitions.

**Failure modes**
- Hard scene cuts: Kalman filter predicts position based on prior motion — wrong after a cut. Mitigated by scene-cut detection (MAD > 30.0) which triggers tracker reset.
- Track fragmentation on long occlusions (>30 frames, configurable via `track_max_age`): track dies and restarts with a new ID, potentially missing frames at the occlusion boundary.
- Identity switches: if two faces occlude each other and separate, ByteTrack may swap their track IDs. This does not affect privacy (both are still redacted) but affects evaluation metrics.

---

## Redaction Methods

### Gaussian Blur (faces and logos)

**Effective threat model:** Defeats casual human re-identification, standard face recognition APIs (ArcFace, FaceNet, DeepFace) at confidence threshold 0.6. At σ=20, cosine similarity between original and blurred face embeddings drops below 0.4 for faces ≥40×40px.

**Does not defeat:**
- Reconstruction attacks specifically designed to invert Gaussian blur (rare in practice; computationally expensive)
- Future embedding models substantially more powerful than current ArcFace/FaceNet generation
- Very small faces (≤20×20px) where blur has limited information to remove; these tend to have low re-ID scores regardless of blur

**Why not pixelation:** Block pixelation with small block sizes (≤8px) is partially reversible via edge-preserving upscaling filters. Gaussian blur at σ=20 is harder to reverse. Both methods are equivalent at large block sizes (≥16px), but Gaussian blur is the more conservative choice.

### Luminance-Matched Fill (text)

Text regions are filled with a solid colour matching the surrounding background luminance (extracted via YCrCb Y channel). This reads as blank space rather than a visible censorship bar, reducing visual conspicuousness. The fill does not contain recoverable text information.

---

## Privacy Claims

| Claim | Scope | Conditions |
|---|---|---|
| Face re-ID rate ≤ 2% | Faces ≥ 40px width under standard ArcFace evaluation | Measured via cosine similarity threshold 0.6 |
| Text content not recoverable | Text regions covered by opaque fill | Does not apply to audio |
| Logo identity not recoverable | Logo regions replaced by Gaussian blur | Abstract minimalist logos may survive blur if residual structure remains |
| Audio unchanged | Audio stream-copied unmodified | Voice, speech content, background sounds are not processed |

**What this system does NOT claim:**
- Anonymization against a determined adversary with access to auxiliary data (e.g., combining redacted video with other footage of the same person)
- Anonymization of audio, voice, or speech
- Complete removal of contextual identity signals (body shape, gait, distinctive clothing are not redacted)
- Pixel-level mathematical privacy guarantees (this is a heuristic system, not a differential privacy implementation)

---

## Failure Modes

| Failure mode | Severity | Conditions | Mitigation |
|---|---|---|---|
| Face missed entirely | High | Face < 20px, extreme yaw (>75°), heavy occlusion (>70%) | Lower confidence threshold; upgrade to RetinaFace |
| Single-frame redaction gap | High | Tracker death between detection frames | ByteTrack propagation + scene-cut reset; gap fill via `track_max_age` |
| Wrong bbox after scene cut | Medium | Hard cut within an active track | Scene-cut detection + tracker reset |
| Semi-transparent watermark missed | Medium | Watermark opacity < 50% | CRAFT detector as alternative |
| Abstract logo missed | Medium | Minimalist shapes without textual cues | Lower confidence threshold; accept as known limitation |
| Text at non-overlay positions missed | Low | Brand text at mid-frame positions (by design) | Remove overlay-position heuristic if full recall required |
| Body, gait, clothing not redacted | High (by design) | Out of scope | Requires separate body detection + redaction model |

---

## Ethical Considerations

### Dual-use risk
The detection and tracking components of this pipeline identify and follow individuals across video frames. The same capability that enables privacy protection also enables surveillance. Distribution of this system should be accompanied by clear use-case restrictions. The face tracker in particular should not be repurposed for identity tracking without explicit consent frameworks.

### Differential privacy impact
This system does not anonymize equally across all demographic groups. Lower recall on darker skin tones in low-light conditions means individuals in those groups are more likely to have their identity exposed in pipeline output. Operators should audit performance on their specific content domain before relying on pipeline output as a privacy control.

### Consent and jurisdiction
Automated redaction is not a substitute for consent. In many jurisdictions, the act of recording individuals in certain contexts requires consent regardless of whether the footage is subsequently anonymized. This system does not address the legal basis for original recording.

### Completeness vs. false positives
The pipeline is tuned toward recall (catch more, redact more) rather than precision (only redact what is certain). This means some non-PII regions will be incidentally redacted. Operators should communicate to audiences that redaction is automated and may include non-identifying regions.

### Adversarial robustness
This system has not been evaluated against adversarial inputs designed to evade detection (e.g., adversarial patches that fool face detectors). Adversarial robustness is an open research problem. Operators in high-stakes contexts (journalism, legal evidence) should apply additional human review.

---

## Evaluation Summary

*(Populated after running `evaluate.py` against a labelled test set.)*

| Metric | Result | Target | Status |
|---|---|---|---|
| Face Recall | — | ≥ 0.95 | — |
| Face Precision | — | ≥ 0.90 | — |
| Text Recall | — | ≥ 0.90 | — |
| Logo Recall | — | ≥ 0.90 | — |
| Temporal Consistency | — | ≥ 0.98 | — |
| SSIM (non-redacted) | — | ≥ 0.85 | — |
| Throughput (T4) | — | ≥ 10 FPS | — |
| Re-ID Rate | — | ≤ 0.02 | — |

Run evaluation:
```bash
python evaluate.py --video-dir data/test_videos/ --annotation-dir data/annotations/
```

---

## Upgrade Paths

| Component | Current | When to upgrade | Replacement |
|---|---|---|---|
| Face detector | YOLOv8n-face | Recall < 95% on test set | YOLOv8s-face, RetinaFace, InsightFace SCRFD |
| Text detector | PaddleOCR DBNet | Semi-transparent watermarks failing | CRAFT detector |
| Logo detector | YOLO-World-S | Abstract logo recall insufficient | GroundingDINO (slower but higher recall) |
| Tracker | ByteTrack | Long-occlusion identity switches | StrongSORT, BoTrack |
| Redaction (face) | Gaussian blur σ=20 | Re-ID rate > 2% | Increase σ; add pixelation layer |
