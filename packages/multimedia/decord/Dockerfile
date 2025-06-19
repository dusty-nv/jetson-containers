#---
# name: decord2
# group: multimedia
# config: config.py
# depends: [numpy, ffmpeg, cmake, video-codec-sdk]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/johnnynunez/decord2
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DECORD_VERSION \
    CUDAARCHS

COPY build.sh install.sh /tmp/decord/

RUN /tmp/decord/install.sh || /tmp/decord/build.sh
