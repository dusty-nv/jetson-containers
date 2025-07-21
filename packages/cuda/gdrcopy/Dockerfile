#---
# name: gdrcopy
# group: cuda
# depends: [cuda]
# config: config.py
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GDRCOPY_VERSION \
    DISTRO="ubuntu2004"

# https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/

RUN apt-get update && \
    apt-get install -y dkms

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    devscripts \
    debhelper \
    fakeroot \
    pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY install.sh /tmp/gdrcopy/
RUN /tmp/gdrcopy/install.sh
