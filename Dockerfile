# Model-aware 3D Eye Gaze
# CUDA 11.8 + PyTorch 2.0.1 + Python 3.10

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    git \
    wget \
    ca-certificates \
    ninja-build \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libegl1-mesa \
    libegl1-mesa-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && ln -sf /usr/bin/python3 /usr/bin/python

ENV FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.5"

WORKDIR /workspace

# Large CUDA/PyTorch packages are installed from local wheels first.
COPY docker_wheels /tmp/wheels

RUN pip install --no-cache-dir --find-links=/tmp/wheels \
    /tmp/wheels/torch-2.0.1+cu118-cp310-cp310-linux_x86_64.whl \
    /tmp/wheels/torchvision-0.15.2+cu118-cp310-cp310-linux_x86_64.whl \
    /tmp/wheels/torchaudio-2.0.2+cu118-cp310-cp310-linux_x86_64.whl

COPY requirements.txt .
RUN pip install --no-cache-dir --find-links=/tmp/wheels -r requirements.txt

RUN pip install --no-cache-dir --find-links=/tmp/wheels \
    /tmp/wheels/pytorch3d-0.7.4-cp310-cp310-linux_x86_64.whl \
    && rm -rf /tmp/wheels

COPY . .

CMD ["python", "run.py", "--help"]
