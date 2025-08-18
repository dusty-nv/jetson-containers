#---
# name: xformers
# group: attention
# config: config.py
# depends: [pytorch, triton, diffusers, flash-attention]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/facebookresearch/xformers
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG XFORMERS_VERSION \
    IS_SBSA \
    FORCE_BUILD=off

ENV XFORMERS_FORCE_DISABLE_TRITON=1 \
    XFORMERS_IGNORE_FLASH_VERSION_CHECK=1

COPY build.sh install.sh /tmp/xformers/

RUN /tmp/xformers/install.sh || /tmp/xformers/build.sh
