#!/usr/bin/env bash
set -ex

# apt dependencies for NERF/3DGS stack
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build pkg-config \
  python3-dev pybind11-dev \
  libeigen3-dev libsuitesparse-dev \
  libgoogle-glog-dev libgflags-dev \
  libflann-dev libfreeimage-dev libmetis-dev \
  libhdf5-dev libboost-filesystem-dev libboost-graph-dev \
  libboost-program-options-dev libboost-system-dev libboost-test-dev \
  libqt5opengl5-dev qtbase5-dev libsqlite3-dev \
  libcgal-dev protobuf-compiler libprotobuf-dev \
  libopenblas-dev liblapack-dev \
  libgtest-dev \
  libceres-dev \
  sudo vim-tiny

export CMAKE_PREFIX_PATH="/usr/lib/aarch64-linux-gnu/cmake:${CMAKE_PREFIX_PATH}"
export glog_DIR="/usr/lib/aarch64-linux-gnu/cmake/glog"
export gflags_DIR="/usr/lib/aarch64-linux-gnu/cmake/gflags"
# Ceres config is usually here on 24.04:
export Ceres_DIR="/usr/lib/aarch64-linux-gnu/cmake/Ceres"

rm -rf /var/lib/apt/lists/*
apt-get clean

rm -rf /var/lib/apt/lists/*
apt-get clean

# either install or build pyceres wheel
if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of pyceres ${PYCERES}"
	exit 1
fi

uv pip install pyceres==${PYCERES_VERSION}
