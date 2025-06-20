#---
# name: xattention
# group: attention
# config: config.py
# depends: [pytorch, torchvision, torchaudio, torchao, flash-attention, transformers, diffusers, opencv, block_sparse_attn]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG XATTENTION_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/xattention/

RUN /tmp/xattention/install.sh || /tmp/xattention/build.sh
