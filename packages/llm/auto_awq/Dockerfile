#---
# name: auto_awq
# group: llm
# config: config.py
# depends: [transformers]
# requires: '>=36'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AUTOAWQ_VERSION \
    AUTOAWQ_KERNELS_VERSION \
    COMPUTE_CAPABILITIES \
    FORCE_BUILD=off

COPY *.sh /tmp/auto_awq/

RUN /tmp/auto_awq/install.sh || /tmp/auto_awq/build.sh
