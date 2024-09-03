#---
# name: videomambasuite
# group: mamba
# config: config.py
# depends: [ninja, pytorch, torchvision, torchaudio, triton, transformers]
# requires: '>=34.1.0'
# notes: https://github.com/OpenGVLab/video-mamba-suite
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VIDEOMAMBASUITE_VERSION \
    CUDAARCHS

COPY build.sh /tmp/VIDEOMAMBASUITE/

RUN pip3 install build wheel && \
    /tmp/VIDEOMAMBASUITE/build.sh
