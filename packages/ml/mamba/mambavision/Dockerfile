#---
# name: mambavision
# group: mamba
# config: config.py
# depends: [mamba]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/OpenGVLab/video-mamba-suite
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MAMBAVISION_VERSION \
    CUDAARCHS

COPY build.sh /tmp/MAMBAVISION/

RUN uv pip install build wheel && \
    /tmp/MAMBAVISION/build.sh
