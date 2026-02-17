#---
# name: tinycudann
# group: nerf
# config: config.py
# depends: [pytorch, torchvision]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/NVlabs/tiny-cuda-nn
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TINYCUDANN_VERSION \
    FORCE_BUILD=off

ENV TCNN_CUDA_ARCHITECTURES=${CUDAARCHS}

RUN git clone --depth=1 --recursive https://github.com/NVlabs/tiny-cuda-nn /opt/tinycudann

COPY build.sh install.sh /tmp/TINYCUDANN/

RUN /tmp/TINYCUDANN/install.sh || /tmp/TINYCUDANN/build.sh
