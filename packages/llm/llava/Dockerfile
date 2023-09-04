#---
# name: llava
# group: llm
# docs: docs.md
# depends: [pytorch, transformers]
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
    sed 's|"deepspeed==0.9.5",.*||' -i pyproject.toml && \
    sed 's|"transformers.*"|"transformers"|' -i pyproject.toml && \
    sed 's|"accelerate.*"|"accelerate"|' -i pyproject.toml && \
    sed 's|"bitsandbytes.*"|"bitsandbytes"|' -i pyproject.toml && \
    cat pyproject.toml

RUN cd llava && \
    pip3 wheel --wheel-dir=dist --no-deps --verbose . && \
    cp dist/llava*.whl /opt
    
RUN pip3 install --no-cache-dir --verbose llava*.whl

WORKDIR /