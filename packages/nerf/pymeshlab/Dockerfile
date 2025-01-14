#---
# name: pymeshlab
# group: nerf
# config: config.py
# depends: [python, cmake, ninja, numpy, polyscope]
# requires: '>=34.1.0'
# notes: https://github.com/cnr-isti-vclab/PyMeshLab.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYMESHLAB_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/PYMESHLAB/
RUN /tmp/PYMESHLAB/install.sh || /tmp/PYMESHLAB/build.sh

