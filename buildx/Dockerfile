# NVIDIA Jetson ML Development Container
# This Dockerfile creates a comprehensive machine learning container for NVIDIA Jetson platforms.
# It includes various ML frameworks, libraries, and utilities optimized for the Jetson platform.

# Build args for the base image
ARG BASE_IMAGE

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
# Install DeepSpeed for distributed training
RUN pip3 install deepspeed

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

# Set environment variables for GPU access
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# ================================
# == Build Completion Marker    ==
# ================================
# Indicate successful build with timestamp
RUN echo "Build completed successfully on 2025-03-30 11:40:54" && \
    echo "Full AI Stack image ready for use"

# ================================
# == Container Configuration    ==
# ================================
# Expose ports for services (Jupyter, TensorBoard, API servers)
EXPOSE 8888 6006 8080

# Default command when starting the container
CMD ["bash"]
