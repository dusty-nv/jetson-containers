#---
# name: polyscope
# group: nerf
# config: config.py
# depends: [cmake, python, numpy]
# requires: '>=34.1.0'
# notes: https://github.com/nmwsharp/polyscope-py.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG POLYSCOPE_VERSION \
    FORCE_BUILD=off

    RUN apt-get update && \
    apt-get install -y --no-install-recommends \ 
    xorg-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev libglfw3 xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh /tmp/POLYSCOPE/

RUN /tmp/POLYSCOPE/install.sh || /tmp/POLYSCOPE/build.sh
    