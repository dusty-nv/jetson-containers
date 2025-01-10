#---
# name: transformer-engine
# group: ml
# config: config.py
# depends: [pytorch, flash-attention]
# requires: '>=34.1.0'
# test: test.py
# notes: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TRANSFORMER_ENGINE_VERSION \
    CUDAARCHS

# COPY build.sh install.sh /tmp/TRANSFORMER_ENGINE/

# COPY patches/${TRANSFORMER_ENGINE_VERSION}.diff /tmp/TRANSFORMER_ENGINE/patch.diff


RUN /tmp/TRANSFORMER_ENGINE/install.sh || /tmp/TRANSFORMER_ENGINE/build.sh
