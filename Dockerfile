# Omnilingual ASR - Docker setup for RTX 3090/4090
FROM nvidia/cuda:12.4-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install PyTorch first (CUDA 12.4)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install fairseq2 (Meta's sequence modeling library)
RUN pip install --no-cache-dir \
    fairseq2==0.5.2 \
    --extra-index-url https://fair.pkg.meta.com/fairseq2/pt2.5.1/cu124

# Install omnilingual-asr
RUN pip install --no-cache-dir omnilingual-asr

# Copy inference scripts
COPY inference.py .
COPY server.py .

# Model cache directory
ENV FAIRSEQ2_CACHE=/models
VOLUME /models

# Default command
CMD ["python", "inference.py", "--help"]
