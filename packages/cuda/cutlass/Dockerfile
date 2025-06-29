#---
# name: cutlass
# group: cuda
# config: config.py
# requires: '>=34.1.0'
# depends: [cuda, numpy, cuda-python]
# test: [test_runtime.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUTLASS_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/cutlass/

RUN /tmp/cutlass/install.sh || /tmp/cutlass/build.sh
