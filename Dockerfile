# =============================================================================
# BraTS 2023 — Fast Conditional Wavelet Diffusion Model (Fast-CWDM) Training
#
# Replicates: https://github.com/tsereda/brats-synthesis
# Data layout on PVC:
#   /pvc/data/brats23/train/BraTS-GLI-*/
#   /pvc/data/brats23/val/BraTS-GLI-*/
#   /pvc/data/brats23/test/BraTS-GLI-*/
#
# Build:
#   docker build -t ghcr.io/kagozi/brats23-cwdm:latest .
#
# Run (example):
#   docker run --gpus all \
#     -v /pvc:/pvc \
#     -e WANDB_API_KEY=<key> \
#     -e WANDB_ENTITY=<entity> \
#     -e TRAIN_MODALITY=t1n \
#     ghcr.io/kagozi/brats23-cwdm:latest
# =============================================================================

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PYTHONPATH=/app

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source from this repo
COPY app/ /app/
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

ENTRYPOINT ["/app/run.sh"]
