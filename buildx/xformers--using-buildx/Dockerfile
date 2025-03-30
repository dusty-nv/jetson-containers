#---
# name: xformers
# group: llm
# config: config.py
# depends: [pytorch]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/facebookresearch/xformers
#---
ARG BASE_IMAGE=kairin/001:2025-03-25-0404
FROM ${BASE_IMAGE}

ARG XFORMERS_VERSION \
    FORCE_BUILD=off

ENV XFORMERS_FORCE_DISABLE_TRITON=1

COPY build.sh install.sh /tmp/xformers/

RUN /tmp/xformers/install.sh || /tmp/xformers/build.sh
    
    
