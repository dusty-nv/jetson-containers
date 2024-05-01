#---
# name: awq
# group: llm
# config: config.py
# depends: [transformers]
# requires: '>=34.1.0'
# test: test.sh
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AWQ_BRANCH \
    AWQ_VERSION \
    AWQ_KERNEL_VERSION

COPY build.sh /tmp/build_awq.sh

RUN pip3 install --no-cache-dir --verbose 'lm-eval<=0.3.0' && \
    pip3 install --no-cache-dir --verbose awq==${AWQ_VERSION} awq-inference-engine==${AWQ_KERNEL_VERSION} || \
    /tmp/build_awq.sh
    