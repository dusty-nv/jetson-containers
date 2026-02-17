#---
# name: pycolmap
# group: nerf
# config: config.py
# depends: [torch, torchvision, ffmpeg, ninja, cmake, pyceres]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/colmap/colmap.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYCOLMAP_VERSION \
    FORCE_BUILD=off

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    wget && \
    rm -rf /var/lib/apt/lists/*


COPY build.sh install.sh /tmp/PYCOLMAP/

RUN cd /opt/
RUN uv pip install build wheel

RUN /tmp/PYCOLMAP/install.sh || /tmp/PYCOLMAP/build.sh
