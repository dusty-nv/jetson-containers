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
    AUTOGPTQ_BRANCH \
    FORCE_BUILD="off"

COPY build.sh /tmp/build_auto_gptq.sh

RUN pip3 install gekko && \
    pip3 install auto-gptq==${AUTOGPTQ_VERSION} || \
    /tmp/build_auto_gptq.sh
