#---
# name: triattention
# group: attention
# config: config.py
# depends: [pytorch, triton, flash-attention, transformers]
# requires: '>=35'
# test: test.py
# notes: https://github.com/WeianMao/triattention.git
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TRIATTENTION_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/triattention/

RUN /tmp/triattention/install.sh || /tmp/triattention/build.sh
