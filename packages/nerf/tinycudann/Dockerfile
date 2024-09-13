#---
# name: tinycudann
# group: nerf
# config: config.py
# depends: [pytorch, torchvision]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/cnr-isti-vclab/TINYCUDANN.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TINYCUDANNN_VERSION \
    FORCE_BUILD=off

ARG CUDAARCHS
# Copy the build and install scripts to the container

ENV TCNN_CUDA_ARCHITECTURES=${CUDAARCHS}

# Clone the tinycudann repository
RUN git clone --depth=1 --recursive https://github.com/NVlabs/tiny-cuda-nn /opt/tinycudann

COPY build.sh install.sh /tmp/TINYCUDANN/

RUN /tmp/TINYCUDANN/install.sh || /tmp/TINYCUDANN/build.sh
