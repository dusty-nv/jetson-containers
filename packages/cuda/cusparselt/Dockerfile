#---
# name: cusparselt
# group: cuda
# depends: [cuda]
# config: config.py
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUSPARSELT_VERSION \
    CUDA_ARCH \
    IS_SBSA \
    FORCE_BUILD=off \
    DISTRO="ubuntu2004" \
    TMP=/tmp/cusparselt

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        devscripts \
        debhelper \
        fakeroot \
        pkg-config \
        dkms && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY install.sh $TMP/install.sh

RUN $TMP/install.sh
