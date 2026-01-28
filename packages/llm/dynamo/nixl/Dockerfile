#---
# name: nixl
# group: dynamo
# config: config.py
# depends: [cuda, rust, cmake, ninja, cudastack, pytorch, torchvision, torchaudio, torchao, transformers, triton, mooncake, go]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/ai-dynamo/nixl
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NIXL_VERSION \
    NIXL_VERSION_SPEC \
    IS_TEGRA \
    IS_SBSA

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    ninja-build \
    pybind11-dev \
    libclang-dev \
    cmake \
    libgflags-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    libaio-dev \
    liburing-dev \
    protobuf-compiler-grpc \
    libcpprest-dev \
    etcd-server \
    etcd-client \
    autotools-dev \
    automake \
    libtool \
    libz-dev \
    flex \
    libgtest-dev \
    build-essential


RUN DEBIAN_FRONTEND=noninteractive apt-get -y install \
    --reinstall libibverbs-dev rdma-core ibverbs-utils libibumad-dev \
    libnuma-dev librdmacm-dev ibverbs-providers

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh /tmp/nixl/

RUN /tmp/nixl/install.sh || /tmp/nixl/build.sh
