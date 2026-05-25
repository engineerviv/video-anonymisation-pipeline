# Target: linux/amd64 + CUDA 11.8 (T4 / A100 benchmark platform)
# For local M1 development use environment.yml instead.
#
# Build:  docker build -t video-anon .
# Run:    docker run --gpus all -v $(pwd)/outputs:/app/outputs video-anon --url <url>
# Eval:   docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs \
#           --entrypoint python video-anon evaluate.py --video-dir data/test_videos/
#
# Models are NOT baked into the image — they download to /app/models on first run.
# Mount a host directory to persist them across runs:
#   -v /host/path/models:/app/models

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# System dependencies
# ffmpeg: video decode/encode (libx264 included)
# libgl1-mesa-glx + libglib2.0: required by OpenCV
# libsm6 + libxext6: required by OpenCV on headless Linux (no display server)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies from requirements.txt (CPU builds as defaults)
# Then swap CPU-only packages for GPU-accelerated equivalents:
#   paddlepaddle     → paddlepaddle-gpu  (PaddleOCR CUDA inference)
#   onnxruntime      → onnxruntime-gpu   (InsightFace ArcFace CUDA inference)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y paddlepaddle onnxruntime \
    && pip install --no-cache-dir "paddlepaddle-gpu>=2.6.0,<3.0.0" "onnxruntime-gpu>=1.17.0"

# Copy project source
COPY pipeline/ ./pipeline/
COPY evaluation/ ./evaluation/
COPY anonymise.py evaluate.py ./

# Create runtime directories
# outputs/ and models/ are typically bind-mounted from the host
RUN mkdir -p models outputs .tmp data/test_videos data/annotations

# PYTHONUNBUFFERED: ensures tqdm progress bars and print() flush immediately
# PYTHONDONTWRITEBYTECODE: keeps the container filesystem clean
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Default: run the anonymization pipeline
# Override with --entrypoint python to run evaluate.py or other scripts
ENTRYPOINT ["python", "anonymise.py"]
