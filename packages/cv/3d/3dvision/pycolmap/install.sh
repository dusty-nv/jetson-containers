#!/usr/bin/env bash
set -euxo pipefail

echo "Installing pycolmap ${PYCOLMAP_VERSION}"

# Install dependencies
apt-get update && \
apt-get install -y --no-install-recommends --no-install-suggests \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-program-options1.83.0 \
    libeigen3-dev \
    libopenblas-dev \
    libomp5 \
    libopenimageio-dev \
    libopenimageio2.4t64 \
    openimageio-tools \
    libopenexr-dev \
    libglew-dev \
    libglew2.2 \
    libgl1 \
    libopengl0 \
    libmetis-dev \
    libmetis5 \
    libceres-dev \
    libceres4t64 \
    libsuitesparse-dev \
    libcgal-dev \
    libgoogle-glog-dev \
    libgoogle-glog0v6t64 \
    libgtest-dev \
    libgmock-dev \
    qt6-base-dev \
    libqt6core6 \
    libqt6gui6 \
    libqt6widgets6 \
    libqt6opengl6-dev \
    libqt6openglwidgets6 \
    libsqlite3-dev \
    libcurl4-openssl-dev \
    libcurl4 \
    libssl-dev \
    libssl3t64 \
    libc6 \
    libgcc-s1 \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

mkdir -p /usr/include/opencv4

if [ "$FORCE_BUILD" == "on" ]; then
  echo "Forcing build of pycolmap ${PYCOLMAP_VERSION}"
  exit 1
fi

PYCOLMAP_TARPACK_NAME="${PYCOLMAP_TARPACK_NAME:-colmap-${PYCOLMAP_VERSION}}"

tarpack install "colmap-${PYCOLMAP_VERSION}" || {echo "tarpack install failed for colmap-${PYCOLMAP_VERSION}, falling back to pip."}

# Fallback general (o arquitecturas no-tegra): instala desde PyPI
uv pip install "pycolmap==${PYCOLMAP_VERSION}"
