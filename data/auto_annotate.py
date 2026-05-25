"""
Auto-annotate real test videos using detectors at low threshold.

Samples every SAMPLE_RATE frames, runs face/text/logo detectors at reduced
thresholds to maximize recall (acting as a proxy ground-truth oracle),
and writes annotation JSON files.

These are proxy annotations, not manual labels. They are biased toward
what the detectors can find but at lower confidence than the pipeline uses.
This is noted in the benchmark report.

Run: python data/auto_annotate.py
"""

import json
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from pipeline.config import PipelineConfig, get_device
from pipeline.detection.face import FaceDetector
from pipeline.detection.text import TextDetector
from pipeline.detection.logo import LogoDetector

VIDEOS = Path("data/test_videos")
ANNOTS = Path("data/annotations")
ANNOTS.mkdir(exist_ok=True)

# Real videos to annotate (skip synthetics — those already have perfect GT)
REAL_VIDEOS = [
    "Broadcast-News-Clip-360p.mp4",
    "Crowd-Footage-Clip.mp4",
    "Dashcam-Clip.mp4",
    "Talking-Head-Clip-1080p.mp4",
    "Vertical-UGC-Clip.mp4",
]

SAMPLE_RATE = 1    # annotate every frame — required for fair IoU matching in evaluate.py
MAX_FRAMES  = 300  # first 300 frames (~10 seconds at 30fps)


def annotate_video(stem: str) -> int:
    vid_path = VIDEOS / stem
    ann_path = ANNOTS / f"{Path(stem).stem}.json"

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open {stem}")
        return 0

    config = PipelineConfig(device=get_device())
    # Lower thresholds than pipeline defaults — maximise recall for proxy GT
    config.face_confidence = 0.25
    config.text_confidence = 0.30
    config.logo_confidence = 0.20

    face_det = FaceDetector(config)
    text_det = TextDetector(config)
    logo_det = LogoDetector(config)

    face_det._ensure_loaded()
    text_det._ensure_loaded()
    logo_det._ensure_loaded()

    annotations = []
    frame_idx = 0
    sampled = 0

    while sampled < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_RATE == 0:
            dets = (
                face_det.detect(frame, frame_idx) +
                text_det.detect(frame, frame_idx) +
                logo_det.detect(frame, frame_idx)
            )
            for d in dets:
                annotations.append({
                    "frame_idx": frame_idx,
                    "bbox": list(d.bbox),
                    "class": d.class_name,
                    "track_id": None,
                })
            sampled += 1

        frame_idx += 1

    cap.release()

    with open(ann_path, "w") as f:
        json.dump({"annotations": annotations}, f, indent=2)

    print(f"  {stem}: {frame_idx} frames sampled every {SAMPLE_RATE} → {len(annotations)} annotations → {ann_path.name}")
    return len(annotations)


if __name__ == "__main__":
    print("Auto-annotating real test videos (proxy GT at low threshold)...")
    total = 0
    for stem in REAL_VIDEOS:
        total += annotate_video(stem)
    print(f"\nDone. {total} total proxy annotations written.")
