#---
# name: cutlass
# group: cuda
# config: config.py
# requires: '>=34.1.0'
# depends: [cuda, numpy, cuda-python, pytorch]
# test: [test_runtime.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUTLASS_VERSION \
    FORCE_BUILD=off

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		clang cmake ninja-build pkg-config libgtk-3-dev liblzma-dev libstdc++-12-dev && \
    rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY build.sh install.sh /tmp/cutlass/

RUN /tmp/cutlass/install.sh || /tmp/cutlass/build.sh
