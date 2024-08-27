#---
# name: pymeshlab
# group: nerf
# config: config.py
# depends: [cmake, python, numpy, polyscope]
# requires: '>=34.1.0'
# notes: https://github.com/cnr-isti-vclab/PyMeshLab.git
# test: test.py
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYMESHLAB_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/PYMESHLAB/

RUN /tmp/PYMESHLAB/install.sh || /tmp/PYMESHLAB/build.sh
    