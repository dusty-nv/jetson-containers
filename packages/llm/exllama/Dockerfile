#---
# name: exllama
# group: llm
# config: config.py
# depends: [pytorch, huggingface_hub, flash-attention, transformers, xformers, diffusers]
# requires: '>=34.1.0'
# test: test.sh
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG EXLLAMA_VERSION \
    EXLLAMA_BRANCH \
    FORCE_BUILD=off

COPY install.sh build.sh /tmp/exllama/

RUN /tmp/exllama/install.sh || /tmp/exllama/build.sh
