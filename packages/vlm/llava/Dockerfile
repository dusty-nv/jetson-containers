#---
# name: llava
# group: vlm
# docs: docs.md
# depends: [pytorch, torchvision, flash-attention, transformers]
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
    sed -i 's/==/>=/g' pyproject.toml && \
    sed -i 's/~=/>=/g' pyproject.toml && \
    cat pyproject.toml

COPY benchmark.py llava/llava/serve/

RUN cd llava && \
    uv build --wheel --out-dir dist --no-deps --verbose . && \
    cp dist/llava*.whl /opt

RUN uv pip install llava*.whl

#RUN uv pip install /opt/torch*.whl
#RUN uv pip install /opt/torchvision*.whl

WORKDIR /opt/llava
