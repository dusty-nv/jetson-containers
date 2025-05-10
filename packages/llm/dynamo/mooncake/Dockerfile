#---
# name: mooncake
# group: dynamo
# config: config.py
# depends: [cuda, rust, cmake, ninja, pytorch, torchvision, torchaudio, torchao, transformers, triton]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/kvcache-ai/Mooncake
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MOONCAKE_VERSION \
    MOONCAKE_VERSION_SPEC \
    IS_SBSA

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        cmake && \
        # etcd etcd-server etcd-client && \ (not in ubuntu 24.04)
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh /tmp/mooncake/

RUN /tmp/mooncake/install.sh || /tmp/mooncake/build.sh
