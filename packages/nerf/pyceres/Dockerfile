#---
# name: pyceres
# group: nerf
# config: config.py
# depends: [numpy, ninja, cmake, opengl]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/cvg/pyceres.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYCERES_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/PYCERES/

RUN /tmp/PYCERES/install.sh || /tmp/PYCERES/build.sh
