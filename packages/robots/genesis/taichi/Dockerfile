#---
# name: taichi
# group: genesis
# config: config.py
# depends: [ninja, cmake, rust]
# requires: '>=36.0.0'
# test: test.py
# notes: https://github.com/taichi-dev/taichi
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TAICHI_VERSION \
    CUDAARCHS

RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    git \
    curl \
    build-essential \
    gcc \
    g++ \
    clang \
    wget \
    bash-completion \
    libgl1 \
    libgl1-mesa-glx \
    libegl-dev \
    libegl1 \
    libxrender1 \
    libglib2.0-0 \
    ffmpeg \
    libgtk2.0-dev \
    pkg-config \
    vulkan-tools \
    libvulkan1 \
    libvulkan-dev \
    libgles2 \
    libglvnd0 \
    libglx0 \
    freeglut3-dev \
    libglfw3-dev \
    libglm-dev \
    libglu1-mesa-dev \
    liblz4-dev \
    libssl-dev \
    libtinfo-dev \
    libwayland-dev \
    libx11-xcb-dev \
    libxcb-dri3-dev \
    libxcb-ewmh-dev \
    libxcb-keysyms1-dev \
    libxcb-randr0-dev \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev \
    libzstd-dev \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*


COPY build.sh install.sh /tmp/TAICHI/

RUN /tmp/TAICHI/install.sh || /tmp/TAICHI/build.sh
