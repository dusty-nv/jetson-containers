#---
# name: sage-attention
# group: attention
# config: config.py
# depends: [pytorch, torchvision, torchaudio, torchao, flash-attention, transformers, diffusers, opencv]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SAGE_ATTENTION_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh patches /tmp/sage-attention/

RUN /tmp/sage-attention/install.sh || /tmp/sage-attention/build.sh
