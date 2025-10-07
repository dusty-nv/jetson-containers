#---
# name: gptqmodel
# alias: gptqmodel
# group: llm
# config: config.py
# depends: [transformers]
# requires: '>=34.1.0'
# test: test.py
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GPTQMODEL_VERSION \
    GPTQMODEL_BRANCH \
    TORCH_CUDA_ARCH_LIST \
    FORCE_BUILD="off"

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
COPY build.sh build.sh /tmp/gptqmodel/

RUN uv pip install gekko && \
    uv pip install auto-gptq==${GPTQMODEL_VERSION} || \
    /tmp/gptqmodel/build.sh
