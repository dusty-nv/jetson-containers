#---
# name: open3d
# alias: open3d
# group: cv
# config: config.py
# depends: [pytorch, torchvision, torchaudio, torchao, opencv]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPEN3D_VERSION \
    TMP_DIR=/tmp/open3d \
    FORCE_BUILD=off

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
    llvm-17 \
    clang-17 \
    libc++-17-dev \
    lld \
    libc++abi-17-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV SUDO=command
COPY build.sh install.sh "${TMP_DIR}"/
RUN "${TMP_DIR}"/install.sh || "${TMP_DIR}"/build.sh || touch "${TMP_DIR}"/.build.failed

# ---- Environment variables ---------------------------------------------------
# 1) XDG_SESSION_TYPE                       (forces X11 inside the container)
ENV XDG_SESSION_TYPE=x11

# 2) LD_PRELOAD — combine both libraries in *one* variable (colon-separated)
#    – first: libgomp (OpenMP, shipped by libgomp1)
#    – second: libOpen3D.so from the venv you just created
# ENV LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libgomp.so.1:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/open3d/cpu/libOpen3D.so.0.19"
