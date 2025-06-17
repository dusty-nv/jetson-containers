#---
# name: causalconv1d
# group: mamba
# config: config.py
# depends: [ninja, pytorch, torchvision, torchaudio, triton]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/Dao-AILab/causal-conv1d
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CASUALCONV1D_VERSION \
    CASUALCONV1D_VERSION_SPEC

COPY build.sh install.sh patches /tmp/causalconv1d/

RUN /tmp/causalconv1d/install.sh || /tmp/causalconv1d/build.sh
