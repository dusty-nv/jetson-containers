#!/usr/bin/env bash
set -euxo pipefail

echo "Installing pycolmap ${PYCOLMAP_VERSION}"

apt-get update && \
apt-get install -y --no-install-recommends \
git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libopenimageio-dev \
    openimageio-tools \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qt6-base-dev \
    libqt6opengl6-dev \
    libqt6openglwidgets6 \
    libcgal-dev \
    libceres-dev \
    libsuitesparse-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libopenblas-dev \
    libopenexr-dev \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean
mkdir -p /usr/include/opencv4
if [ "$FORCE_BUILD" == "on" ]; then
  echo "Forcing build of pycolmap ${PYCOLMAP_VERSION}"
  exit 1
fi

PYCOLMAP_TARPACK_NAME="${PYCOLMAP_TARPACK_NAME:-colmap-${PYCOLMAP_VERSION}}"

tarpack install "colmap-${PYCOLMAP_VERSION}" || {echo "tarpack install failed for colmap-${PYCOLMAP_VERSION}, falling back to pip."}

# Fallback general (o arquitecturas no-tegra): instala desde PyPI
uv pip install "pycolmap==${PYCOLMAP_VERSION}"
