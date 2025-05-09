#---
# name: gdrcopy
# group: cuda
# depends: [cuda]
# config: config.py
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GDRCOPY_VERSION

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

RUN git clone --recursive https://github.com/NVIDIA/gdrcopy.git /opt/gdrcopy

RUN cd /opt/gdrcopy/packages && \
    sed -i 's/gdrdrv-dkms (= @FULL_VERSION@), //g' debian-meta/control && \
    CUDA=/usr/local/cuda ./build-deb-packages.sh && \
    mkdir debs && \
    mv *.deb debs && \
    rm -rf debs/gdrdrv-dkms*.deb && \
    ls -la debs && \
    dpkg -i debs/*.deb && \
    rm -rf /opt/gdrcopy && \
    ldconfig
