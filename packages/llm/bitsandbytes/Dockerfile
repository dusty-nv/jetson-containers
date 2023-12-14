#---
# name: bitsandbytes
# group: llm
# requires: '>=35.*'
# depends: [transformers]
# test: test.py
# notes: fork of https://github.com/TimDettmers/bitsandbytes for Jetson
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# upstream version is https://github.com/TimDettmers/bitsandbytes (the fork below includes some patches for Jetson) 
ARG BITSANDBYTES_REPO=jasl/bitsandbytes
ARG BITSANDBYTES_BRANCH=jetson

# force rebuild on new git commits - https://stackoverflow.com/a/56945508
ADD https://api.github.com/repos/${BITSANDBYTES_REPO}/git/refs/heads/${BITSANDBYTES_BRANCH} /tmp/bitsandbytes_version.json

WORKDIR /opt

RUN git clone --branch=${BITSANDBYTES_BRANCH} --depth=1 https://github.com/${BITSANDBYTES_REPO} bitsandbytes
RUN cd bitsandbytes && \
    CUDA_VERSION=122 make -j$(nproc) cuda12x && \
    CUDA_VERSION=122 make -j$(nproc) cuda12x_nomatmul && \
    pip3 wheel --wheel-dir=dist --no-deps --verbose . && \
    cp dist/bitsandbytes*.whl /opt

RUN pip3 install --no-cache-dir --verbose --no-build-isolation /opt/bitsandbytes*.whl

RUN pip3 install --no-cache-dir --verbose scipy
RUN pip3 show bitsandbytes && python3 -c 'import bitsandbytes'
