#---
# name: mlstm_kernels
# group: xlstm
# config: config.py
# depends: [cuda, pytorch, triton, flash-attention]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/NX-AI/mlstm_kernels
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MLSTM_KERNELS_VERSION \
    MLSTM_KERNELS_VERSION_SPEC \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/mlstm_kernels/
RUN /tmp/mlstm_kernels/install.sh || /tmp/mlstm_kernels/build.sh
