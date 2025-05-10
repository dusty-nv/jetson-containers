#---
# name: xgrammar
# group: llm
# config: config.py
# depends: [cmake, ninja, pytorch, torchvision, torchaudio, transformers]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/mlc-ai/xgrammar.git
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG XGRAMMAR_VERSION \
    CUDAARCHS \
    IS_SBSA \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/xgrammar/

RUN /tmp/xgrammar/install.sh || /tmp/xgrammar/build.sh
RUN /tmp/transformers/install.sh
