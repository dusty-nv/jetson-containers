#!/usr/bin/env bash
set -ex
apt-get update -y && \
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
  pkg-config \
  build-essential && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of nixl ${NIXL_VERSION}"
	exit 1
fi

uv pip install "nixl~=${NIXL_VERSION}" || \
uv pip install "nixl~=${NIXL_VERSION_SPEC}"
