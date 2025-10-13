#---
# name: isaaclab
# alias: isaaclab
# group: sim
# config: config.py
# depends: [ffmpeg, cmake, h5py, pytorch, torchvision, torchaudio, transformers, isaacsim]
# requires: '>=35'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ISAACLAB_VERSION \
    FORCE_BUILD="off"

COPY build.sh install.sh /tmp/isaaclab/

RUN /tmp/isaaclab/install.sh || /tmp/isaaclab/build.sh || echo "BUILD FAILED (ISAACLAB ${ISAACLAB_VERSION})"

WORKDIR /opt
