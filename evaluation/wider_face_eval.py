"""
WIDER FACE evaluation — honest face recall/precision against human-annotated GT.

Why this matters:
  Our proxy evaluation (auto_annotate.py) uses the same YOLOv8n-face model at a
  lower confidence threshold to generate ground truth. That produces self-consistent
  but circular numbers — it measures how well we agree with ourselves, not how well
  we detect real faces.

  WIDER FACE is a public benchmark with 32,203 images and 393,703 human-annotated
  face bounding boxes across extreme conditions (scale, pose, occlusion, lighting).
  YOLOv8n-face was TRAINED on WIDER FACE, so this is an in-distribution recall test.
  It answers: "On the hardest face detection benchmark, how often does our detector
  fire on a real face?"

Setup (on Kaggle):
  1. Add the WIDER FACE dataset to your notebook:
     Kaggle > "Add Data" > search "wider face" > add "lukyanov/wider-face"
  2. Run:
     python -m evaluation.wider_face_eval \\
       --ann  /kaggle/input/wider-face/wider_face_split/wider_face_val_bbx_gt.txt \\
       --imgs /kaggle/input/wider-face/WIDER_val/images

Setup (local):
  Download WIDER FACE validation split from http://shuoyang1213.me/WIDERFACE/
  and point --ann / --imgs at the extracted directories.

Usage:
    python -m evaluation.wider_face_eval --ann <path> --imgs <path>
    python -m evaluation.wider_face_eval --ann <path> --imgs <path> --max 500
    python -m evaluation.wider_face_eval --ann <path> --imgs <path> --full

Output:
    Prints recall / precision / F1 table.
    Saves results to outputs/wider_face_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ── Dataset reader ─────────────────────────────────────────────────────────────

def _parse_wider_face_annotations(ann_path: Path) -> list[dict]:
    """
    Parse wider_face_val_bbx_gt.txt annotation file.

    Format:
        <relative/image/path.jpg>
        <num_faces>
        x1 y1 w h blur expression illumination invalid occlusion pose
        ... (one line per face, repeated num_faces times)
        (if num_faces == 0, a single "0 0 0 0 0 0 0 0 0 0" placeholder follows)

    Returns list of dicts: {image_rel: str, boxes: list[(x1,y1,x2,y2)]}
    where boxes are in absolute pixel coords, already converted from xywh.
    """
    records = []

    with open(ann_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        image_rel = lines[i]
        i += 1

        if i >= len(lines):
            break

        num_faces = int(lines[i])
        i += 1

        boxes = []
        face_count = max(num_faces, 1)  # placeholder line even when 0 faces
        for _ in range(face_count):
            if i >= len(lines):
                break
            parts = lines[i].split()
            i += 1
            if len(parts) < 4:
                continue
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            if num_faces > 0 and w > 0 and h > 0:
                boxes.append((x, y, x + w, y + h))

        records.append({"image_rel": image_rel, "boxes": boxes})

    return records


# ── IoU and matching ───────────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _match(pred_boxes: list, gt_boxes: list, iou_thr: float) -> tuple[int, int, int]:
    """Returns (tp, fp, fn) via greedy highest-IoU matching."""
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            v = _iou(pb, gb)
            if v >= iou_thr:
                pairs.append((v, pi, gi))
    pairs.sort(reverse=True)

    matched_pred, matched_gt = set(), set()
    for _, pi, gi in pairs:
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)

    tp = len(matched_gt)
    return tp, len(pred_boxes) - len(matched_pred), len(gt_boxes) - len(matched_gt)


# ── Main evaluation ────────────────────────────────────────────────────────────

def run_evaluation(
    ann_path: Path,
    images_dir: Path,
    max_images: int | None = 500,
    iou_threshold: float = 0.5,
    face_confidence: float = 0.45,
    min_face_px: int = 8,
) -> dict:
    """
    Evaluate face detector against WIDER FACE validation annotations.

    Args:
        ann_path:        Path to wider_face_val_bbx_gt.txt
        images_dir:      Path to WIDER_val/images/
        max_images:      Limit evaluation to first N images. None = full set.
        iou_threshold:   IoU threshold for TP (standard: 0.5).
        face_confidence: Detection conf — matches live pipeline default (0.45).
        min_face_px:     Skip GT faces smaller than this. Assignment min = 8px.
    """
    from pipeline.config import PipelineConfig, get_device
    from pipeline.detection.face import FaceDetector

    print("=" * 60)
    print("WIDER FACE Evaluation")
    print(f"  conf={face_confidence}  IoU≥{iou_threshold}  min_face={min_face_px}px")
    n_label = f"first {max_images}" if max_images else "all"
    print(f"  evaluating {n_label} validation images")
    print("=" * 60)

    print("\nLoading face detector...")
    device = get_device()
    config = PipelineConfig(device=device, face_confidence=face_confidence)
    detector = FaceDetector(config)
    detector._ensure_loaded()
    print(f"  Device: {device}")

    print("\nParsing annotations...")
    records = _parse_wider_face_annotations(ann_path)
    print(f"  {len(records)} annotated images in validation set")
    if max_images:
        records = records[:max_images]

    import cv2

    total_tp = total_fp = total_fn = 0
    images_done = skipped_missing = skipped_no_gt = 0
    start = time.perf_counter()

    for rec in records:
        img_path = images_dir / rec["image_rel"]
        if not img_path.exists():
            skipped_missing += 1
            continue

        # Filter GT boxes by minimum size
        gt_boxes = [
            b for b in rec["boxes"]
            if (b[2] - b[0]) >= min_face_px and (b[3] - b[1]) >= min_face_px
        ]

        if not gt_boxes:
            skipped_no_gt += 1
            images_done += 1
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            skipped_missing += 1
            continue

        detections = detector.detect(img_bgr, frame_idx=images_done)
        pred_boxes = [d.bbox for d in detections]

        tp, fp, fn = _match(pred_boxes, gt_boxes, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        images_done += 1

        if images_done % 50 == 0:
            elapsed = time.perf_counter() - start
            r = total_tp / max(total_tp + total_fn, 1)
            p = total_tp / max(total_tp + total_fp, 1)
            print(
                f"  [{images_done:4d}] recall={r:.3f}  prec={p:.3f}  "
                f"tp={total_tp} fp={total_fp} fn={total_fn}  ({elapsed:.0f}s)"
            )

    elapsed = time.perf_counter() - start
    total_gt = total_tp + total_fn
    total_pred = total_tp + total_fp

    recall = total_tp / max(total_gt, 1)
    precision = total_tp / max(total_pred, 1)
    f1 = 2 * recall * precision / max(recall + precision, 1e-9)

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "total_gt_faces": total_gt,
        "total_pred_faces": total_pred,
        "images_evaluated": images_done,
        "images_skipped_missing": skipped_missing,
        "images_skipped_no_gt": skipped_no_gt,
        "elapsed_s": round(elapsed, 1),
        "face_confidence": face_confidence,
        "iou_threshold": iou_threshold,
        "min_face_px": min_face_px,
    }


def _print_results(r: dict) -> None:
    print("\n" + "=" * 60)
    print("WIDER FACE Results")
    print("=" * 60)
    print(f"  Images evaluated  : {r['images_evaluated']}")
    print(f"  GT faces (≥{r['min_face_px']}px)   : {r['total_gt_faces']}")
    print(f"  Predictions        : {r['total_pred_faces']}")
    print()
    print(f"  True  positives   : {r['tp']}")
    print(f"  False positives   : {r['fp']}")
    print(f"  False negatives   : {r['fn']}")
    print()
    print(f"  Recall            : {r['recall']:.4f}  ({r['recall']*100:.1f}%)")
    print(f"  Precision         : {r['precision']:.4f}  ({r['precision']*100:.1f}%)")
    print(f"  F1 score          : {r['f1']:.4f}")
    print()
    print(f"  Elapsed           : {r['elapsed_s']:.1f}s")
    print(f"  Settings          : conf={r['face_confidence']}  IoU≥{r['iou_threshold']}")
    print("=" * 60)


def _find_wider_face_paths() -> tuple[Path | None, Path | None]:
    """
    Auto-discover WIDER FACE annotation file and images directory under
    /kaggle/input/. Kaggle dataset slugs vary (wider-face, widerface, etc.)
    so we search by filename rather than assuming a fixed path.
    """
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None, None

    # Find annotation file anywhere under /kaggle/input
    ann = None
    for candidate in kaggle_input.rglob("wider_face_val_bbx_gt.txt"):
        ann = candidate
        break

    # Find images directory: look for WIDER_val/images or any dir named 'images'
    # that sits next to a wider_face_split dir
    imgs = None
    if ann is not None:
        # Annotation is in wider_face_split/; images should be a sibling directory
        split_dir = ann.parent.parent  # up from wider_face_split/
        for candidate in ["WIDER_val/images", "wider_val/images", "val/images", "images"]:
            p = split_dir / candidate
            if p.exists():
                imgs = p
                break

    if imgs is None:
        # Broader search: any 'images' directory that contains subdirs (categories)
        for candidate in kaggle_input.rglob("WIDER_val/images"):
            imgs = candidate
            break

    return ann, imgs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WIDER FACE face detector evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kaggle setup:
  Add dataset: "Add Data" → search "wider face" → add lukyanov/wider-face
  Then run with default paths (no --ann / --imgs needed).

Local setup:
  Download WIDER FACE val split from http://shuoyang1213.me/WIDERFACE/
  Point --ann and --imgs at the extracted files.
""",
    )
    parser.add_argument("--ann", default=None, metavar="PATH",
        help="Path to wider_face_val_bbx_gt.txt (auto-discovered on Kaggle if omitted)")
    parser.add_argument("--imgs", default=None, metavar="PATH",
        help="Path to WIDER_val/images/ directory (auto-discovered on Kaggle if omitted)")
    parser.add_argument("--max", type=int, default=500, metavar="N",
        help="Evaluate first N images (default: 500). Use --full for all.")
    parser.add_argument("--full", action="store_true",
        help="Evaluate full validation set (~3226 images, ~30 min on T4).")
    parser.add_argument("--iou", type=float, default=0.5, metavar="THRESH",
        help="IoU threshold for TP matching (default: 0.5)")
    parser.add_argument("--conf", type=float, default=0.45, metavar="CONF",
        help="Face detection confidence (default: 0.45, same as pipeline)")
    parser.add_argument("--min-face", type=int, default=8, metavar="PX",
        help="Skip GT faces smaller than this in any dim (default: 8)")
    parser.add_argument("--output", default=None,
        help="Save JSON results here (default: outputs/wider_face_results.json)")
    args = parser.parse_args()

    # Resolve annotation and images paths
    if args.ann is None or args.imgs is None:
        auto_ann, auto_imgs = _find_wider_face_paths()
        ann_path  = Path(args.ann)  if args.ann  else auto_ann
        imgs_dir  = Path(args.imgs) if args.imgs else auto_imgs
    else:
        ann_path = Path(args.ann)
        imgs_dir = Path(args.imgs)

    if ann_path is None or not ann_path.exists():
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            print("ERROR: Could not find wider_face_val_bbx_gt.txt under /kaggle/input/")
            print()
            print("The WIDER FACE dataset is not attached to this notebook.")
            print("Fix — in your Kaggle notebook:")
            print("  1. Click 'Add Data' (right sidebar, looks like a + icon)")
            print("  2. Search: wider face a face detection benchmark")
            print("  3. Add:    mksaad/wider-face-a-face-detection-benchmark")
            print("  4. Re-run this cell")
            print()
            print("Top-level dirs under /kaggle/input/ right now:")
            try:
                for p in sorted(kaggle_input.iterdir()):
                    print(f"  {p.name}/")
            except Exception:
                pass
        else:
            print(f"ERROR: Annotation file not found: {ann_path or '(not provided)'}")
            print("Local: download WIDER FACE val split from http://shuoyang1213.me/WIDERFACE/")
            print("       then pass --ann / --imgs pointing at the extracted files.")
        sys.exit(1)

    if imgs_dir is None or not imgs_dir.exists():
        print(f"ERROR: Images directory not found: {imgs_dir}")
        print("Pass --imgs pointing at the WIDER_val/images directory.")
        sys.exit(1)

    print(f"  Annotations : {ann_path}")
    print(f"  Images      : {imgs_dir}")

    max_images = None if args.full else args.max

    results = run_evaluation(
        ann_path=ann_path,
        images_dir=imgs_dir,
        max_images=max_images,
        iou_threshold=args.iou,
        face_confidence=args.conf,
        min_face_px=args.min_face,
    )

    _print_results(results)

    output_path = Path(args.output) if args.output else _ROOT / "outputs" / "wider_face_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {output_path}")


if __name__ == "__main__":
    main()
