#---
# name: block_sparse_attn
# group: attention
# config: config.py
# depends: [pytorch, torchvision, torchaudio, torchao, flash-attention]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BLOCKSPARSEATTN_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/block_sparse_attn/

RUN /tmp/block_sparse_attn/install.sh || /tmp/block_sparse_attn/build.sh
