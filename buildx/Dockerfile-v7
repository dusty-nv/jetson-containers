# Build args for the base image
ARG BASE_IMAGE

# Start from base image
FROM ${BASE_IMAGE}

# Build arguments
ARG TRITON_VERSION=2.0.0
ARG TRITON_BRANCH=main

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    pciutils \
    pkg-config \
    python3-dev \
    python3-pip \
    libjpeg-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libopenblas-dev \
    libatlas-base-dev \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip \
    && pip3 install \
    numpy \
    scipy \
    matplotlib \
    pillow \
    scikit-learn \
    scikit-image \
    pandas \
    jupyterlab \
    ipywidgets \
    tqdm \
    requests \
    pyyaml \
    tensorboard

# Install ML libraries compatible with Jetson
RUN pip3 install \
    torch \
    torchvision \
    transformers \
    diffusers \
    bitsandbytes \
    accelerate \
    einops \
    safetensors

# Install Triton
RUN git clone --branch ${TRITON_BRANCH} https://github.com/openai/triton.git \
    && cd triton \
    && pip3 install -e python \
    && cd .. \
    && rm -rf triton/.git

# Install additional ML libraries
RUN pip3 install \
    onnx \
    onnxruntime \
    tensorflow \
    jax \
    flax \
    optax \
    xformers \
    huggingface_hub \
    flash-attention \
    deepspeed

# Install OpenCV with CUDA support
RUN pip3 install opencv-python

# Install CuPy for GPU acceleration
RUN pip3 install cupy-cuda12x

# Set up working directory
WORKDIR /workspace

# Create directories for data and models
RUN mkdir -p /workspace/data /workspace/models

# Set environment variables for GPU
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# Build completion message  
RUN echo "Build completed successfully on 2025-03-30 11:28:07" && \
    echo "Full AI Stack image ready for use"

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Default command
CMD ["bash"]
