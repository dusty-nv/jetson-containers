#---
# name: glomap
# group: nerf
# config: config.py
# depends: [pycolmap]
# requires: '>=34.1.0'
# notes: https://github.com/colmap/colmap.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GLOMAP_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/GLOMAP/

RUN /tmp/GLOMAP/install.sh || /tmp/GLOMAP/build.sh
