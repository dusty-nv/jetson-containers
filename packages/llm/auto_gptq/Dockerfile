#---
# name: auto_gptq
# alias: auto-gptq
# group: llm
# config: config.py
# depends: [transformers]
# requires: '>=34.1.0'
# test: test.py
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AUTOGPTQ_VERSION \
    AUTOGPTQ_BRANCH

COPY build.sh /tmp/build_auto_gptq.sh

RUN pip3 install --no-cache-dir gekko && \
    pip3 install --no-cache-dir --verbose auto-gptq==${AUTOGPTQ_VERSION} || \
    /tmp/build_auto_gptq.sh
