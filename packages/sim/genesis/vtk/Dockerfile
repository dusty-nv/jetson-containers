#---
# name: vtk
# group: sim
# config: config.py
# depends: [ninja, cmake, rust]
# requires: '>=36.0.0'
# test: test.py
# notes: https://gitlab.kitware.com/vtk/vtk
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VTK_VERSION \
    VTK_VERSION_SPEC \
    CUDAARCHS

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    ninja-build \
    cmake \
    libgl1-mesa-dev \
    python3-dev \
    git \
    git \
    curl \
    clang \
    wget \
    bash-completion \
    libgl1 \
    libglx-mesa0 \
    libegl-dev \
    libegl1 \
    libxrender1 \
    libglib2.0-0 \
    ffmpeg \
    libgtk2.0-dev \
    pkg-config \
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


COPY build.sh install.sh /tmp/vtk/

RUN /tmp/vtk/install.sh || /tmp/vtk/build.sh
