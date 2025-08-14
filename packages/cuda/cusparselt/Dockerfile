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

# Convert ARG to ENV so values are available at runtime for testing
ENV CUSPARSELT_VERSION=$CUSPARSELT_VERSION
ENV CUDA_ARCH=$CUDA_ARCH
ENV IS_SBSA=$IS_SBSA
ENV FORCE_BUILD=$FORCE_BUILD
ENV DISTRO=$DISTRO
ENV TMP=$TMP

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
