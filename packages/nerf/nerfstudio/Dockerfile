#---
# name: nerfstudio
# group: nerf
# config: config.py
# depends: [cuda, python, ninja, cmake, numpy, torch, torchvision, polyscope, pymeshlab]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/nerfstudio-project/nerfstudio.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NERFSTUDIO_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/NERFSTUDIO/

RUN /tmp/NERFSTUDIO/install.sh || /tmp/NERFSTUDIO/build.sh
    