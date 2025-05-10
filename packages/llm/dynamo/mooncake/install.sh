#!/usr/bin/env bash
set -ex
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential devscripts debhelper fakeroot pkg-config \
        cmake etcd etcd-server etcd-client \
        libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
        libcpprest-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of mooncake ${MOONCAKE_VERSION}"
	exit 1
fi
"ai-mooncake[all]"

pip3 install "mooncake~=${MOONCAKE_VERSION}" || \
pip3 install "mooncake~=${MOONCAKE_VERSION_SPEC}"