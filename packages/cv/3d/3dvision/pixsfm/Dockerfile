#---
# name: pixsfm
# group: nerf
# config: config.py
# depends: [torch, torchvision, ffmpeg, python, ninja, cmake, numpy, pyceres, pycolmap, hloc]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/cvg/Hierarchical-Localization.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PIXSFM_VERSION \
FORCE_BUILD=off

ARG CUDAARCHS

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    libgtest-dev \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*


COPY build.sh install.sh /tmp/PIXSFM/

RUN cd /opt/
RUN uv pip install build wheel

RUN /tmp/PIXSFM/install.sh || /tmp/PIXSFM/build.sh
