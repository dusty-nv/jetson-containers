#---
# name: tvm
# group: ml
# depends: [pytorch, rust, llvm:21]
# config: config.py
# requires: '>=34.1.0'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

ARG TVM_VERSION \
    CUDAARCHS \
    TORCH_CUDA_ARCH_LIST \
    FORCE_BUILD=off

# MLC/TVM recommends to use LLVM
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# copy installation and build scripts for TVM
COPY build.sh install.sh /tmp/tvm/

RUN /tmp/tvm/install.sh || /tmp/tvm/build.sh

WORKDIR /

# export TVM envs for downstream stages/users
ENV TVM_HOME=/opt/tvm \
    PYTHONPATH=/opt/tvm/python:${PYTHONPATH}
