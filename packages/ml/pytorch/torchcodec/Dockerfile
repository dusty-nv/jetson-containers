#---
# name: torchcodec
# group: pytorch
# config: config.py
# depends: [video-codec-sdk, ffmpeg, pybind11, pyav, pytorch, cmake]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCHCODEC_VERSION \
    FORCE_BUILD=off

COPY install.sh build.sh /tmp/torchcodec/

RUN /tmp/torchcodec/install.sh || /tmp/torchcodec/build.sh
