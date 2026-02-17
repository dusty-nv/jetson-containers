#---
# name: pymeshlab
# group: nerf
# config: config.py
# depends: [numpy, cmake, ninja, polyscope]
# requires: '>=34.1.0'
# test: test.sh
# notes: https://github.com/cnr-isti-vclab/PyMeshLab.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYMESHLAB_VERSION \
    FORCE_BUILD=off

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    fontconfig \
    fuse \
    gdb \
    git \
    kmod \
    libboost-dev \
    libdbus-1-3 \
    libegl-dev \
    libfuse2 \
    libgmp-dev \
    libglu1-mesa-dev \
    libmpfr-dev \
    libpulse-mainloop-glib0 \
    libtbb-dev \
    libxerces-c-dev \
    patchelf \
    qtbase5-dev \
    qt5-qmake \
    rsync \
    unzip \
    wget \
    libqt5opengl5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh /tmp/PYMESHLAB/

RUN /tmp/PYMESHLAB/install.sh || /tmp/PYMESHLAB/build.sh

