#---
# name: mambavision
# group: mamba
# config: config.py
# depends: [ninja, pytorch, torchvision, torchaudio, triton, transformers, causalconv1d]
# requires: '>=34.1.0'
# notes: https://github.com/OpenGVLab/video-mamba-suite
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MAMBAVIDEO_VERSION \
    CUDAARCHS

COPY build.sh /tmp/MAMBAVIDEO_VERSION/

RUN pip3 install build wheel && \
    /tmp/MAMBAVIDEO_VERSION/build.sh
