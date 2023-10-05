#---
# name: bitsandbytes
# group: llm
# requires: '>=34.1.0'
# depends: [build-essential, pytorch]
# test: test.py
# notes: fork of https://github.com/TimDettmers/bitsandbytes for Jetson
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# upstream version is https://github.com/TimDettmers/bitsandbytes (the fork below includes some patches for Jetson) 
ARG BITSANDBYTES_REPO=dusty-nv/bitsandbytes
ARG BITSANDBYTES_BRANCH=main

# force rebuild on new git commits - https://stackoverflow.com/a/56945508
ADD https://api.github.com/repos/${BITSANDBYTES_REPO}/git/refs/heads/${BITSANDBYTES_BRANCH} /tmp/bitsandbytes_version.json

RUN pip3 uninstall -y bitsandbytes && \
    cd /opt && \
    git clone --depth=1 https://github.com/${BITSANDBYTES_REPO} bitsandbytes && \
    cd bitsandbytes && \
    CUDA_VERSION=114 make -j$(nproc) cuda11x && \
    CUDA_VERSION=114 make -j$(nproc) cuda11x_nomatmul && \
    python3 setup.py --verbose build_ext --inplace -j$(nproc) bdist_wheel && \
    cp dist/bitsandbytes*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/bitsandbytes*.whl  && \
    cd ../ && \
    rm -rf bitsandbytes

RUN pip3 install --no-cache-dir --verbose scipy
RUN pip3 show bitsandbytes && python3 -c 'import bitsandbytes'
