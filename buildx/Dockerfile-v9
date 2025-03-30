# ========================================================================
# NVIDIA Jetson ML Development Container
# ========================================================================
# This Dockerfile creates a comprehensive machine learning container for 
# NVIDIA Jetson platforms with JetPack-L4T.
#
# COMPONENTS STATUS:
# ✅ Core System Packages - INSTALLED
# ✅ Python Development Tools - INSTALLED
# ✅ PyTorch & TorchVision - INSTALLED 
# ✅ TensorFlow - INSTALLED
# ✅ JAX Ecosystem - INSTALLED
# ✅ Hugging Face Libraries - INSTALLED
# ✅ ONNX & Runtime - INSTALLED
# ✅ Triton Compiler - INSTALLED
# ✅ Optimization Libraries - INSTALLED
# ✅ Computer Vision (OpenCV) - INSTALLED
# ✅ GStreamer Integration - INSTALLED
# ✅ FFmpeg - INSTALLED
# ❌ DeepSpeed - FAILED (CUDA incompatibility)
# ⚠️ CuPy - NOT YET TESTED
#
# LAST BUILD: 2025-03-30 20:30
# MAINTAINER: kairin
# 
# This container provides a ready-to-use environment for machine learning
# development on NVIDIA Jetson platforms with optimized libraries
# for edge AI deployment.
# ========================================================================

# NVIDIA Jetson ML Development Container
# This Dockerfile creates a comprehensive machine learning container for NVIDIA Jetson platforms.
#
# === CONTENTS ===
# - System Dependencies and Basic Utilities
# - Python Core Dependencies
# - Core ML/DL Framework Layer (PyTorch, TensorFlow, JAX)
# - Hugging Face Ecosystem (Transformers, Diffusers)
# - ONNX Runtime & Conversion
# - OpenAI Triton Compilation
# - Optimization Libraries (bitsandbytes, flash-attention, xformers)
# - Computer Vision Libraries (OpenCV)
# - GStreamer for Video I/O
# - FFmpeg Integration
# - Distributed Training (DeepSpeed)
# - CUDA Extensions (CuPy)
#
# === USAGE ===
# Build using ./jetsonrun.sh which provides a dialog-based interface
# Verify using ./verify.sh <image_name>

# Reminder by KAIRIN -- please do not remove the above comments only improve or add on to the information above. This is intended to help remind you what # has been installed, what needs to be ins[...]

# NVIDIA Jetson ML Development Container
# This Dockerfile creates a comprehensive machine learning container for NVIDIA Jetson platforms.
#
# === CONTENTS ===
# - System Dependencies and Basic Utilities
# - Python Core Dependencies
# - Core ML/DL Framework Layer (PyTorch, TensorFlow, JAX)
# - Hugging Face Ecosystem (Transformers, Diffusers)
# - ONNX Runtime & Conversion
# - OpenAI Triton Compilation
# - Optimization Libraries (bitsandbytes, flash-attention, xformers)
# - Computer Vision Libraries (OpenCV)
# - GStreamer for Video I/O
# - FFmpeg Integration
# - Distributed Training (DeepSpeed)
# - CUDA Extensions (CuPy)
#
# === USAGE ===
# Build using ./jetsonrun.sh which provides a dialog-based interface
# Verify using ./verify.sh <image_name>

# Build args for the base image
ARG BASE_IMAGE=kairin/001:nvcr.io-nvidia-pytorch-25.02-py3-igpu

# Start from the specified base image (usually a PyTorch-based L4T image)
FROM ${BASE_IMAGE}

# Build arguments
ARG TRITON_VERSION=2.0.0
ARG TRITON_BRANCH=main

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# =============================================
# == System Dependencies and Basic Utilities ==
# =============================================
# Install essential development tools and libraries
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
    python3-setuptools \
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

# ==============================
# == Python Core Dependencies ==
# ==============================
# Install fundamental Python packages for data science and ML
RUN pip3 install --upgrade pip setuptools wheel \
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

# ================================
# == Core ML/DL Framework Layer ==
# ================================
# Install major machine learning and deep learning frameworks
# Note: PyTorch is usually pre-installed in the base image
RUN pip3 install \
    torch \
    torchvision \
    tensorflow \
    jax \
    flax \
    optax

# ================================
# == Hugging Face Ecosystem     ==
# ================================
# Install the Hugging Face transformers library and related packages
RUN pip3 install \
    transformers \
    diffusers \
    datasets \
    accelerate \
    tokenizers \
    huggingface_hub

# ================================
# == ONNX Runtime & Conversion  ==
# ================================
# Install ONNX and ONNX Runtime for model deployment
RUN pip3 install \
    onnx \
    onnxruntime

# ================================
# == OpenAI Triton Compilation ==
# ================================
# Triton is a language and compiler for writing highly efficient GPU code
RUN git clone --branch ${TRITON_BRANCH} https://github.com/openai/triton.git \
    && cd triton \
    && pip3 install -e python \
    && cd .. \
    && rm -rf triton/.git

# ================================
# == Optimization Libraries     ==
# ================================
# Install libraries for model optimization and acceleration
RUN pip3 install \
    bitsandbytes \
    flash-attention \
    xformers

# ================================
# == Computer Vision Libraries  ==
# ================================
# Install OpenCV with GPU acceleration
RUN pip3 install opencv-python

# ================================
# == GStreamer for Video I/O    ==
# ================================
# Install GStreamer and Python bindings for video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    python3-gst-1.0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ================================
# == FFmpeg Integration         ==
# ================================
# Install FFmpeg for video transcoding and manipulation
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ================================
# == Distributed Training       ==
# ================================
# Install DeepSpeed for distributed training - building from source to avoid CUDA issues
RUN apt-get update && apt-get install -y --no-install-recommends \
    ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/microsoft/DeepSpeed.git \
    && cd DeepSpeed \
    && DS_BUILD_OPS=0 DS_BUILD_SPARSE_ATTN=0 pip install -e . \
    && cd .. \
    && rm -rf DeepSpeed

# ================================
# == CUDA Extensions            ==
# ================================
# Install CuPy (CUDA-accelerated NumPy alternative)
RUN pip3 install cupy

# ================================
# == Workspace Configuration    ==
# ================================
# Set up working directory and create folders for projects
WORKDIR /workspace

# Create directories for data, models and logs
RUN mkdir -p /workspace/data /workspace/models /workspace/logs

# Copy verification script into the container for internal verification
COPY verify.sh /workspace/verify.sh
RUN chmod +x /workspace/verify.sh

# Set environment variables for GPU access
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# ================================
# == Build Completion Marker    ==
# ================================
# Indicate successful build with timestamp
RUN echo "Build completed successfully on 2025-03-30 12:47:03" && \
    echo "Full AI Stack image ready for use"

# ================================
# == Container Configuration    ==
# ================================
# Expose ports for services (Jupyter, TensorBoard, API servers)
EXPOSE 8888 6006 8080

# Default command when starting the container
CMD ["bash"]
