#---
# name: ctranslate2
# group: transformer
# config: config.py
# depends: [cuda, cudastack:standard, cmake]
# requires: '>=34.1.0'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CTRANSLATE_VERSION \
    CTRANSLATE_BRANCH \
    CTRANSLATE_SOURCE=/opt/CTranslate2 \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/ctranslate/

RUN /tmp/ctranslate/install.sh || /tmp/ctranslate/build.sh
