#---
# name: pymeshlab
# group: nerf
# config: config.py
# depends: [cmake, ninja, python, numpy, polyscope]
# requires: '>=34.1.0'
# notes: https://github.com/cnr-isti-vclab/PyMeshLab.git
# test: test.py
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYMESHLAB_VERSION \
    FORCE_BUILD=off

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
git \
python3-pip \
libeigen3-dev \
qtbase5-dev \
qtbase5-dev-tools \
libqt5opengl5-dev \
mesa-common-dev \
libglu1-mesa-dev \
make \
patchelf \
libgmp-dev \
libmpfr-dev \
libxerces-c-dev \
libtbb-dev \
libxkbcommon-x11-dev \
libxcb-icccm4-dev \
libxcb-image0-dev \
libxcb-keysyms1-dev \
libxcb-render-util0-dev \
libxcb-xinerama0-dev \
&& rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh /tmp/PYMESHLAB/

RUN /tmp/PYMESHLAB/install.sh || /tmp/PYMESHLAB/build.sh