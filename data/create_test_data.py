"""
Generate synthetic test videos + ground-truth annotation JSON files.

Five synthetic clips, each 60 frames (2s @ 30fps), covering distinct content
domains. Bounding boxes are drawn programmatically so annotations are exact.

Run: python data/create_test_data.py
Outputs:
  data/test_videos/synth_*.mp4
  data/annotations/synth_*.json

Annotation format (matches load_ground_truth() in evaluation/metrics.py):
  { "annotations": [
      {"frame_idx": 0, "bbox": [x1,y1,x2,y2], "class": "face", "track_id": 1},
      ...
  ]}
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

OUT_VIDEOS = Path("data/test_videos")
OUT_ANNOTS = Path("data/annotations")
OUT_VIDEOS.mkdir(parents=True, exist_ok=True)
OUT_ANNOTS.mkdir(parents=True, exist_ok=True)

W, H, FPS, N = 1280, 720, 30, 60  # 2-second clips


def save(stem: str, frames: list[np.ndarray], annotations: list[dict]) -> None:
    vid_path = OUT_VIDEOS / f"{stem}.mp4"
    ann_path = OUT_ANNOTS / f"{stem}.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_path), fourcc, FPS, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()

    with open(ann_path, "w") as fp:
        json.dump({"annotations": annotations}, fp, indent=2)

    print(f"  {vid_path.name}  ({len(annotations)} annotations)")


def draw_face(img, cx, cy, rx=90, ry=110, track_id=1) -> tuple:
    """Draw a synthetic face oval. Returns (x1,y1,x2,y2)."""
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (145, 105, 82), -1)
    cv2.circle(img, (cx - rx//3, cy - ry//5), rx//7, (40, 30, 25), -1)
    cv2.circle(img, (cx + rx//3, cy - ry//5), rx//7, (40, 30, 25), -1)
    cv2.ellipse(img, (cx, cy + ry//3), (rx//2, ry//5), 0, 0, 180, (80, 50, 45), 2)
    return (max(0, cx - rx), max(0, cy - ry), min(W, cx + rx), min(H, cy + ry))


def draw_text_overlay(img, text, y_pos, color=(255, 255, 255)) -> tuple:
    """Draw a lower-third text overlay. Returns bbox."""
    x0, y0 = 30, y_pos - 40
    x1, y1 = W - 30, y_pos + 10
    cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 40), -1)
    cv2.putText(img, text, (x0 + 10, y0 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return (x0, y0, x1, y1)


def draw_logo(img, label, x, y) -> tuple:
    """Draw a simple logo badge. Returns bbox."""
    x1, y1 = x, y
    x2, y2 = x + 180, y + 60
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.putText(img, label, (x1 + 10, y1 + 42),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (30, 30, 180), 2, cv2.LINE_AA)
    return (x1, y1, x2, y2)


# ── Clip 1: Interview / Studio ────────────────────────────────────────────────
def make_synth_interview():
    frames, anns = [], []
    for i in range(N):
        img = np.full((H, W, 3), (50, 45, 60), dtype=np.uint8)
        pan = i * 1
        bbox = draw_face(img, 640 + pan, 340, 100, 120, track_id=1)
        text_bbox = draw_text_overlay(img, "Dr. Jane Smith — Senior Researcher", H - 90)
        logo_bbox = draw_logo(img, "ResearchTV", W - 200, 15)
        anns.append({"frame_idx": i, "bbox": list(bbox), "class": "face", "track_id": 1})
        anns.append({"frame_idx": i, "bbox": list(text_bbox), "class": "text", "track_id": 10})
        anns.append({"frame_idx": i, "bbox": list(logo_bbox), "class": "logo", "track_id": 20})
        frames.append(img)
    save("synth_interview", frames, anns)


# ── Clip 2: News Broadcast ────────────────────────────────────────────────────
def make_synth_news():
    frames, anns = [], []
    for i in range(N):
        img = np.full((H, W, 3), (30, 30, 50), dtype=np.uint8)
        # Anchor face (left)
        b1 = draw_face(img, 300, 350, 90, 110, track_id=1)
        # Reporter face (right, smaller — remote feed)
        b2 = draw_face(img, 950, 250, 55, 65, track_id=2)
        # Lower-third ticker
        t1 = draw_text_overlay(img, "BREAKING: Summit Begins in Geneva", H - 100)
        # Network logo top-right
        l1 = draw_logo(img, "GlobalNews", W - 195, 12)
        for bbox, cls, tid in [(b1,"face",1),(b2,"face",2),(t1,"text",10),(l1,"logo",20)]:
            anns.append({"frame_idx": i, "bbox": list(bbox), "class": cls, "track_id": tid})
        frames.append(img)
    save("synth_news", frames, anns)


# ── Clip 3: Multi-face Panel ──────────────────────────────────────────────────
def make_synth_panel():
    frames, anns = [], []
    positions = [(220, 360, 80, 95), (640, 360, 80, 95), (1060, 360, 80, 95)]
    for i in range(N):
        img = np.full((H, W, 3), (35, 35, 45), dtype=np.uint8)
        for tid, (cx, cy, rx, ry) in enumerate(positions, start=1):
            slight_pan = i // 10 * 2
            bbox = draw_face(img, cx + slight_pan, cy, rx, ry, track_id=tid)
            anns.append({"frame_idx": i, "bbox": list(bbox), "class": "face", "track_id": tid})
        t1 = draw_text_overlay(img, "Live Panel Discussion", H - 90)
        l1 = draw_logo(img, "TalkShow", 15, 15)
        anns.append({"frame_idx": i, "bbox": list(t1), "class": "text", "track_id": 10})
        anns.append({"frame_idx": i, "bbox": list(l1), "class": "logo", "track_id": 20})
        frames.append(img)
    save("synth_panel", frames, anns)


# ── Clip 4: Social / UGC vertical-style (letterboxed) ────────────────────────
def make_synth_ugc():
    frames, anns = [], []
    for i in range(N):
        img = np.full((H, W, 3), (25, 25, 30), dtype=np.uint8)
        # Letterbox bars
        img[:, :280] = (10, 10, 10)
        img[:, 1000:] = (10, 10, 10)
        # Face in centre column
        cx = 640 + int(10 * np.sin(i * 0.3))
        bbox = draw_face(img, cx, 320, 75, 90, track_id=1)
        anns.append({"frame_idx": i, "bbox": list(bbox), "class": "face", "track_id": 1})
        # Social text sticker (bottom of active area)
        t1 = (290, H - 120, 990, H - 60)
        cv2.rectangle(img, t1[:2], t1[2:], (0, 180, 255), -1)
        cv2.putText(img, "@username  #trending #viral", (300, H - 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        anns.append({"frame_idx": i, "bbox": list(t1), "class": "text", "track_id": 10})
        frames.append(img)
    save("synth_ugc", frames, anns)


# ── Clip 5: Street / CCTV style ───────────────────────────────────────────────
def make_synth_cctv():
    frames, anns = [], []
    rng = np.random.default_rng(42)
    for i in range(N):
        # Noisy grey-toned background
        img = rng.integers(30, 70, (H, W, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 1)
        # Two small faces moving across frame
        x1 = 200 + i * 8
        b1 = draw_face(img, x1, 420, 40, 50, track_id=1)
        x2 = 900 - i * 4
        b2 = draw_face(img, x2, 380, 35, 42, track_id=2)
        # CCTV timestamp overlay (top-left)
        ts_bbox = (5, 5, 420, 45)
        cv2.rectangle(img, ts_bbox[:2], ts_bbox[2:], (0, 0, 0), -1)
        cv2.putText(img, f"CAM-04  2024-05-25 14:{i//2:02d}:00",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        for bbox, cls, tid in [(b1,"face",1),(b2,"face",2),(ts_bbox,"text",10)]:
            anns.append({"frame_idx": i, "bbox": list(bbox), "class": cls, "track_id": tid})
        frames.append(img)
    save("synth_cctv", frames, anns)


if __name__ == "__main__":
    print("Generating synthetic test videos...")
    make_synth_interview()
    make_synth_news()
    make_synth_panel()
    make_synth_ugc()
    make_synth_cctv()
    print("Done. 5 synthetic videos + annotations written.")
