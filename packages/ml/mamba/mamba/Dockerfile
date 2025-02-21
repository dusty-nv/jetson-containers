#---
# name: mamba
# group: mamba
# config: config.py
# depends: [transformers, causalconv1d, triton]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/state-spaces/mamba
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MAMBA_VERSION \
    CUDAARCHS

COPY build.sh install.sh /tmp/MAMBA/

COPY patches/${MAMBA_VERSION}.diff /tmp/MAMBA/patch.diff

COPY benchmarks /tmp/MAMBA/benchmarks

RUN /tmp/MAMBA/install.sh || /tmp/MAMBA/build.sh
