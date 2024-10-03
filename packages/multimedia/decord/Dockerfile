#---
# name: decord
# group: multimedia
# config: config.py
# depends: [cuda, cudnn, ffmpeg, python, numpy]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/state-spaces/decord
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DECORD_VERSION \
    CUDAARCHS

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev python3-setuptools make cmake \
    ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get clean

COPY build.sh install.sh /tmp/decord/

RUN /tmp/decord/install.sh || /tmp/decord/build.sh
