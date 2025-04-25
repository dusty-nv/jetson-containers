#---
# name: causalconv1d
# group: mamba
# config: config.py
# depends: [ninja, pytorch, torchvision, torchaudio, triton]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/state-spaces/mamba
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CASUALCONV1D_VERSION \
    CASUALCONV1D_VERSION_SPEC \
    CUDAARCHS

COPY build.sh install.sh patches /tmp/casualconv1d/

RUN /tmp/casualconv1d/install.sh || /tmp/casualconv1d/build.sh
