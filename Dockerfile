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

# PyTorch ecosystem — installed from official index (CUDA 11.8)
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch3D — pre-built wheel for CUDA 11.8 + Python 3.10
RUN pip install --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html

COPY . .

CMD ["python", "run.py", "--help"]
