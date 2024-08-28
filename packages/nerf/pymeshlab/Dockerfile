#---
# name: pymeshlab
# group: nerf
# config: docker-compose.yaml
# depends: [python, cmake, ninja, numpy, polyscope]
# requires: '>=34.1.0'
# notes: https://github.com/cnr-isti-vclab/PyMeshLab.git
# test: test.py
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYMESHLAB_VERSION \
    FORCE_BUILD=off

RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get -y install \
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
libxcb-icccm4-dev \
libxcb-image0-dev \
libxcb-keysyms1-dev \
libxcb-render-util0-dev \
libxcb-shape0 \
libxcb-xinerama0-dev \
libxcb-xkb-dev \
libxkbcommon-x11-dev \
libxerces-c-dev \
patchelf \
rsync \
libeigen3-dev \
qtbase5-dev \
qtbase5-dev-tools \
libqt5opengl5-dev \
mesa-common-dev \
make \
unzip \
wget \
&& rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh /tmp/PYMESHLAB/

RUN mkdir -p /tmp/PYMESHLAB/extra
COPY extra/ /tmp/PYMESHLAB/extra
RUN ls -la /tmp/PYMESHLAB/extra*

RUN /tmp/PYMESHLAB/install.sh || /tmp/PYMESHLAB/build.sh