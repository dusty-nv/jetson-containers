#---
# name: flash_attn
# group: llm
# docs: docs.md
# depends: [pytorch]
# requires: '>=34.1.0'
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FLASH_ATTN_REPO=jasl/flash-attention
ARG FLASH_ATTN_BRANCH=aarch64

ADD https://api.github.com/repos/${FLASH_ATTN_REPO}/git/refs/heads/${FLASH_ATTN_BRANCH} /tmp/flash_attn_version.json

WORKDIR /opt

RUN git clone --branch=${FLASH_ATTN_BRANCH} --depth=1 https://github.com/${FLASH_ATTN_REPO} --recursive flash_attn
RUN cd flash_attn && \
    MAX_JOBS=8 FORCE_BUILD=True CUDA_GENCODE='arch=compute_87,code=sm_87' pip3 wheel --wheel-dir=dist --no-deps --verbose . && \
    cp dist/flash_attn*.whl /opt
    
RUN pip3 install --no-cache-dir --verbose --no-build-isolation flash_attn*.whl
