#---
# name: decord
# group: multimedia
# config: config.py
# depends: [numpy, ffmpeg, cmake]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/dmlc/decord
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DECORD_VERSION \
    CUDAARCHS

COPY build.sh install.sh /tmp/decord/

RUN /tmp/decord/install.sh || /tmp/decord/build.sh
