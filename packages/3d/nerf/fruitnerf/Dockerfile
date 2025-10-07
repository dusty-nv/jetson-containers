#---
# name: fruitnerf
# group: nerf
# config: config.py
# depends: [nerfstudio, opencv, onnxruntime]
# requires: '>=34.1.0'
# notes: https://github.com/meyerls/FruitNeRF
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FRUITNERF_VERSION \
    CUDAARCHS

COPY build.sh /tmp/FRUITNERF/

RUN uv pip install build wheel && \
    /tmp/FRUITNERF/build.sh
