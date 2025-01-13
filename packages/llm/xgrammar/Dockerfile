#---
# name: xgrammar
# group: xgrammar
# config: config.py
# depends: [torchaudio, triton]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/mlc-ai/xgrammar
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG XGRAMMAR_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/xgrammar/

RUN /tmp/xgrammar/install.sh || /tmp/xgrammar/build.sh
