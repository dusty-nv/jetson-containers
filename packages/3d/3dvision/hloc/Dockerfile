#---
# name: hloc
# group: nerf
# config: config.py
# depends: [pycolmap, opencv]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/cvg/Hierarchical-Localization.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HLOC_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/HLOC/

RUN /tmp/HLOC/install.sh || /tmp/HLOC/build.sh
