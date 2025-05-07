#---
# name: open3d
# alias: open3d
# group: cv
# config: config.py
# depends: [cuda, cudnn, python, numpy, opengl, vulkan, video-codec-sdk, ffmpeg, pytorch, torchvision, torchaudio, pyav, pycuda, cmake, ninja, llvm:20, opencv]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPEN3D_VERSION \
    TMP_DIR=/tmp/open3d

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    lsb-release \
    pkg-config \
    gnupg \
    git \
    git-lfs \
    gdb \
    wget \
    wget2 \
    curl \
    nano \
    zip \
    unzip \
    time \
    sshpass \
    ssh-client \
    ninja-build \
    gfortran \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    build-essential \
    cmake \
    git \
    gdb \
    libeigen3-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libosmesa6-dev \
    libpng-dev \
    lxde \
    mesa-utils \
    x11vnc \
    xorg-dev \
    xterm \
    xvfb \
    ne \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh "${TMP_DIR}"/
RUN "${TMP_DIR}"/install.sh || "${TMP_DIR}"/build.sh || touch "${TMP_DIR}"/.build.failed