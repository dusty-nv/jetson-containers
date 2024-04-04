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
    AUTOAWQ_CUDA_ARCH

COPY build.sh /tmp/build_autoawq.sh

RUN pip3 install --no-cache-dir --verbose autoawq-kernels==${AUTOAWQ_KERNELS_VERSION} autoawq==${AUTOAWQ_VERSION} || \
    /tmp/build_autoawq.sh
