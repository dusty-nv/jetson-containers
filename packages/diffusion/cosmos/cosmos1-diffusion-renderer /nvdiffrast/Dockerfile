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

COPY build.sh install.sh /tmp/nvdiffrast/

RUN /tmp/nvdiffrast/install.sh || /tmp/nvdiffrast/build.sh


