#---
# name: nixl
# group: distributed
# config: config.py
# depends: [cuda, rust, cmake, ninja, gdrcopy, pytorch, torchvision, torchaudio, torchao, transformers, triton, mooncake]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/ai-dynamo/nixl
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NIXL_VERSION \
    NIXL_VERSION_SPEC \
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
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh /tmp/nixl/

RUN /tmp/nixl/install.sh || /tmp/nixl/build.sh

ENV NIXL_PREFIX=/usr/local/nixl

RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
        x86_64) PLUGIN_ARCH_DIR=x86_64-linux-gnu ;; \
        aarch64) PLUGIN_ARCH_DIR=aarch64-linux-gnu ;; \
        *) echo "Unsupported architecture: $ARCH" && exit 1 ;; \
    esac && \
    echo "/usr/local/nixl/lib/$PLUGIN_ARCH_DIR" > /etc/ld.so.conf.d/nixl.conf && \
    echo "/usr/local/nixl/lib/$PLUGIN_ARCH_DIR/plugins" >> /etc/ld.so.conf.d/nixl.conf && \
    echo "export NIXL_PLUGIN_DIR=/usr/local/nixl/lib/$PLUGIN_ARCH_DIR/plugins" > /etc/profile.d/nixl.sh && \
    ldconfig

# Default to x86_64 path — overridden at runtime by profile.d script if needed
ENV NIXL_PLUGIN_DIR=/usr/local/nixl/lib/x86_64-linux-gnu/plugins