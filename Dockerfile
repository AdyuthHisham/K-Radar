FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

LABEL author="Adyuth" \
      version="kradar-cuda11.3-py38-torch1.12" \
      description="K-Radar env: CUDA 11.3 + PyTorch 1.12.1+cu113 + Python 3.8.13 (Miniconda)"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    KRADAR_ROOT=/opt/K-Radar \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -yq \
    wget unzip build-essential g++ gcc \
    libgl1-mesa-glx libglib2.0-0 \
    openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y sshfs fuse3 && rm -rf /var/lib/apt/lists/*

# Install Miniconda (Python 3.8)
RUN cd /tmp && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda clean -afy

# Install PyTorch 1.12.1+cu113
RUN conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch && \
    conda clean -afy

# Install Python dependencies
RUN pip install --no-cache-dir \
    open3d==0.15.2 \
    easydict \
    tensorboard \
    opencv-python==4.2.0.32 \
    spconv-cu113 \
    nms \
    setuptools==59.5.0 \
    PyQt5 \
    scikit-image \
    numba \
    einops \
    SharedArray \
    tqdm \
    scipy

# Create directory (will be volume-mounted)
RUN mkdir -p /opt/K-Radar

# Set working directory
WORKDIR /opt/K-Radar

# Default command
CMD ["/bin/bash"]
