#---
# name: llava
# group: vlm
# docs: docs.md
# depends: [pytorch, flash-attention, transformers]
# requires: '>=34.1.0'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

ARG LLAVA_REPO=haotian-liu/LLaVA
ARG LLAVA_BRANCH=main

ADD https://api.github.com/repos/${LLAVA_REPO}/git/refs/heads/${LLAVA_BRANCH} /tmp/llava_version.json

RUN git clone --branch=${LLAVA_BRANCH} --depth=1 https://github.com/${LLAVA_REPO} llava && \
    cd llava && \
    sed 's|torch==.*"|torch"|' -i pyproject.toml && \
    sed 's|"bitsandbytes.*",||' -i pyproject.toml && \
    cat pyproject.toml

COPY benchmark.py llava/llava/serve/

RUN cd llava && \
    pip3 wheel --wheel-dir=dist --no-deps --verbose . && \
    cp dist/llava*.whl /opt
    
RUN pip3 install --no-cache-dir --verbose llava*.whl

#RUN pip3 install --verbose /opt/torch*.whl
#RUN pip3 install --verbose /opt/torchvision*.whl

WORKDIR /
