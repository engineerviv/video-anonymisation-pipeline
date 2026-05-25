#!/usr/bin/env bash
# One-shot Kaggle T4 environment setup.
#
# Run at the top of your Kaggle notebook:
#   !git clone https://github.com/engineerviv/video-anonymisation-pipeline.git
#   %cd video-anonymisation-pipeline
#   !bash setup_kaggle.sh
#
# Kaggle already has: torch, torchvision, numpy, opencv-python, Pillow, tqdm
# We install the delta packages, pin supervision, and swap CPU builds → GPU.
#
# IMPORTANT — version pins (do not loosen without testing):
#   paddleocr <3.0.0  : PaddleOCR 3.x removed use_gpu, ocr(rec=False), and other
#                       kwargs the text detector depends on. Breaks silently on 3.x.
#   supervision <0.30 : ByteTrack was removed from supervision 0.30. Tracker breaks.

set -euo pipefail

echo "=== video-anonymisation-pipeline: Kaggle T4 setup ==="

pip install -q \
    "yt-dlp>=2024.1.0" \
    "av>=11.0.0" \
    "ffmpeg-python>=0.2.0" \
    "ultralytics>=8.1.0" \
    "paddleocr>=2.7.3,<3.0.0" \
    "supervision>=0.21.0,<0.30.0" \
    "scikit-image>=0.22.0" \
    "insightface>=0.7.3" \
    "click>=8.1.0"

# Swap CPU PaddlePaddle → GPU build (paddleocr<3.0 requires paddlepaddle 2.x)
pip uninstall -q -y paddlepaddle 2>/dev/null || true
pip install -q "paddlepaddle-gpu>=2.6.0,<3.0.0"

# Swap CPU onnxruntime → GPU build (needed by insightface ArcFace)
pip uninstall -q -y onnxruntime 2>/dev/null || true
pip install -q "onnxruntime-gpu>=1.17.0"

echo ""
echo "=== Setup complete ==="
echo "Anonymize:  python anonymise.py --url <url>"
echo "Evaluate:   python evaluate.py --video-dir data/test_videos/ --annotation-dir data/annotations/"
