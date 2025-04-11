#---
# name: isaac-lab
# alias: isaaclab
# group: sim
# config: config.py
# depends: [ffmpeg, cmake, h5py, pytorch, torchvision, torchaudio, transformers]
# requires: '>=35'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ISAACLAB_VERSION \
    ISAACLAB_BRANCH \
    FORCE_BUILD="off"

COPY build.sh install.sh /tmp/ISAACLAB/

RUN /tmp/ISAACLAB/install.sh || /tmp/ISAACLAB/build.sh || echo "BUILD FAILED (ISAACLAB ${ISAACLAB_VERSION})"

