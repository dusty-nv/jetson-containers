#---
# name: mamba
# group: llm
# config: config.py
# depends: [causalconv1d, triton, transformers]
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


RUN /tmp/MAMBA/install.sh || /tmp/MAMBA/build.sh
