#---
# name: nvdiffrast
# group: diffusion
# config: config.py
# depends: [pytorch, torchvision, transformers, onnx]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/NVlabs/nvdiffrast.git
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NVDIFFRAST_VERSION \
    FORCE_BUILD=off

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY build.sh install.sh /tmp/nvdiffrast/

RUN /tmp/nvdiffrast/install.sh || /tmp/nvdiffrast/build.sh
