#---
# name: polyscope
# group: nerf
# config: config.py
# depends: [numpy, cmake, opengl]
# requires: '>=34.1.0'
# notes: https://github.com/nmwsharp/polyscope-py.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG POLYSCOPE_VERSION \
    POLYSCOPE_VERSION_SPEC \
    FORCE_BUILD=off

RUN apt-get update && \
    apt-get install -y --no-install-recommends xorg-dev xvfb && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

COPY build.sh install.sh /tmp/POLYSCOPE/

RUN /tmp/POLYSCOPE/install.sh || /tmp/POLYSCOPE/build.sh

