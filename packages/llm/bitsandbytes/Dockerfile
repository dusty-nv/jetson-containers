#---
# name: bitsandbytes
# group: llm
# config: config.py
# depends: [build-essential, pytorch]
# test: test.py
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
    git clone --depth=1 https://github.com/dusty-nv/bitsandbytes /opt/bitsandbytes && \
    cd bitsandbytes && \
    CUDA_VERSION=114 make cuda11x && \
    pip3 install --verbose .

RUN pip3 install --no-cache-dir --verbose scipy
RUN pip3 show bitsandbytes && python3 -c 'import bitsandbytes'
