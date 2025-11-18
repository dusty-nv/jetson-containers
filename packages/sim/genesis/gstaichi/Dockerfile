#---
# name: gstaichi
# group: sim
# config: config.py
# depends: [cuda, ninja, cmake, rust, llvm:21, vulkan]
# requires: '>=36.0.0'
# test: test.py
# notes: https://github.com/Genesis-Embodied-AI/gstaichi
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GSTAICHI_VERSION \
    CUDAARCHS

# Clean and update apt
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update

# Install dependencies
RUN apt-get install -y --no-install-recommends \
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
    libncurses-dev \
    libncurses5-dev \
    libncursesw5-dev \
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
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    clang-20 \
    libc++-dev \
    libc++abi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clean apt cache and update package lists
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update

# Install required packages, including libtinfo-dev and ncurses variants
RUN apt-get update && apt-get install -y --no-install-recommends \
    libncurses-dev \
    libtinfo-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh /tmp/gstaichi/

RUN /tmp/gstaichi/install.sh || /tmp/gstaichi/build.sh
