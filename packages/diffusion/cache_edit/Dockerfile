#---
# name: cache_dit
# group: diffusion
# config: config.py
# depends: [cuda, pytorch, triton, transformers, diffusers]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/vipshop/cache-dit
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CACHE_DIT_VERSION \
    CACHE_DIT_VERSION_SPEC

COPY build.sh install.sh /tmp/cache_dit/
RUN /tmp/cache_dit/install.sh || /tmp/cache_dit/build.sh
