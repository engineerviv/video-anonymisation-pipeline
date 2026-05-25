#!/usr/bin/env bash
# One-shot Kaggle T4 environment setup.
#
# Run at the top of your Kaggle notebook:
#   !git clone https://github.com/<you>/video-anon-take-home.git
#   %cd video-anon-take-home
#   !bash setup_kaggle.sh
#
# Kaggle already has: torch, torchvision, numpy, opencv-python, Pillow, tqdm
# We install the delta packages, pin supervision, and swap paddlepaddle CPU → GPU.
#
# IMPORTANT — supervision pin:
#   supervision>=0.30 removed the ByteTrack API we depend on.
#   The pin <0.30 is not optional — remove it and the tracker breaks silently.

set -euo pipefail

echo "=== video-anon: Kaggle T4 setup ==="

pip install -q \
    "yt-dlp>=2024.1.0" \
    "av>=11.0.0" \
    "ffmpeg-python>=0.2.0" \
    "ultralytics>=8.1.0" \
    "paddleocr>=2.7.3" \
    "supervision>=0.21.0,<0.30.0" \
    "scikit-image>=0.22.0" \
    "insightface>=0.7.3" \
    "onnxruntime-gpu>=1.17.0" \
    "click>=8.1.0"

# Swap CPU PaddlePaddle for GPU build
# Suppress errors if paddlepaddle was never installed in this environment
pip uninstall -q -y paddlepaddle 2>/dev/null || true
pip install -q "paddlepaddle-gpu>=2.6.0"

echo ""
echo "=== Setup complete ==="
echo "Anonymize:  python anonymise.py --url <url>"
echo "Evaluate:   python evaluate.py --video-dir data/test_videos/ --annotation-dir data/annotations/"
