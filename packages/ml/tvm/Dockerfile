#---
# name: tvm
# group: ml
# depends: [pytorch, rust]
# config: config.py
# requires: '>=34.1.0'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

ARG CUDAARCHS
ARG TORCH_CUDA_ARCH_LIST

ARG TVM_COMMIT
ARG TVM_VERSION
ARG FORCE_BUILD=on


# MLC/TVM recommends to use LLVM
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		llvm-18 \
		llvm-18-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
# copy installation and build scripts for TVM
COPY install.sh build.sh /tmp/tvm/

RUN chmod +x /tmp/tvm/install.sh /tmp/tvm/build.sh

# attempt to install TVM from pip, and fall back to building it from source
RUN /tmp/tvm/install.sh || /tmp/tvm/build.sh

WORKDIR /

# export TVM envs for downstream stages/users
ENV TVM_HOME=/opt/tvm \
    TVM_SOURCE_DIR=/opt/tvm \
    PYTHONPATH=/opt/tvm/python:${PYTHONPATH}