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
    CUDAARCHS

COPY build.sh install.sh /tmp/CASUALCONV1D/

COPY patches/${CASUALCONV1D_VERSION}.diff /tmp/CASUALCONV1D/patch.diff


RUN /tmp/CASUALCONV1D/install.sh || /tmp/CASUALCONV1D/build.sh
