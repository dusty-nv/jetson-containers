#---
# name: torchao
# group: pytorch
# config: config.py
# depends: [pytorch, triton, cmake]
# buildkit_device: nvidia.com/gpu=all
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCHAO_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/torchao/

RUN /tmp/torchao/install.sh || /tmp/torchao/build.sh
