#!/bin/bash

set -ex

INSTALL_ARCH=$(uname -m)
if [ -z "${INSTALL_ARCH}" ]; then
    echo "no architecture detected"
    exit 1
fi

apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    ca-certificates \
    git \
    gcc-10 \
    g++-10

if [ -n "${CMAKE_VERSION}" ]; then
    curl -s -L https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${INSTALL_ARCH}.tar.gz | tar -zx -C /usr --strip-components 1
fi

if [ -n "${GOLANG_VERSION}" ]; then
    GO_ARCH="arm64"
    curl -s -L https://dl.google.com/go/go${GOLANG_VERSION}.linux-${GO_ARCH}.tar.gz | tar xz -C /usr/local
    ln -s /usr/local/go/bin/go /usr/local/bin/go
    ln -s /usr/local/go/bin/gofmt /usr/local/bin/gofmt
fi

rm -rf /var/lib/apt/lists/* && apt-get clean
