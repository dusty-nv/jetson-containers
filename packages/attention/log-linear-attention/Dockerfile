#---
# name: log-linear-attention
# group: attention
# config: config.py
# depends: [pytorch, triton, xformers, mamba]
# buildkit_device: nvidia.com/gpu=all
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG LOG_LINEAR_ATTN_VERSION \
    IS_SBSA \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/log-linear-attention/

RUN /tmp/log-linear-attention/install.sh || /tmp/log-linear-attention/build.sh
    
