#---
# name: tensorrt
# group: cuda
# depends: [cuda, cudnn, python]
# config: config.py
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TENSORRT_URL
ARG TENSORRT_DEB
ARG TENSORRT_PACKAGES

RUN echo "Downloading ${TENSORRT_DEB}" && \
    mkdir /tmp/cudnn && cd /tmp/cudnn && \
    wget --quiet --show-progress --progress=bar:force:noscroll ${TENSORRT_URL} && \
    dpkg -i *.deb && \
    cp /var/nv-tensorrt-local-repo-*/nv-tensorrt-local-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-cache search tensorrt && \
    apt-get install -y --no-install-recommends ${TENSORRT_PACKAGES} && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    dpkg --list | grep tensorrt && \
    dpkg -P ${TENSORRT_DEB} && \
    rm -rf /tmp/tensorrt