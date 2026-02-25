#---
# name: tensorrt_edgellm
# group: llm
# config: config.py
# depends: [cuda, cudastack:standard, cmake, python, numpy, pytorch, torchvision, transformers, nvidia_modelopt]
# requires: '>=36'
# test: test.py
# notes: https://github.com/NVIDIA/TensorRT-Edge-LLM
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TENSORRT_EDGELLM_VERSION \
    TENSORRT_EDGELLM_BRANCH \
    CUDA_ARCHITECTURES \
    FORCE_BUILD=off \
    SOURCE_DIR=/opt/TensorRT-Edge-LLM

ADD https://api.github.com/repos/NVIDIA/TensorRT-Edge-LLM/git/refs/heads/main /tmp/tensorrt_edgellm_version.json

COPY install.sh build.sh /tmp/tensorrt_edgellm/

RUN /tmp/tensorrt_edgellm/install.sh || /tmp/tensorrt_edgellm/build.sh

RUN if [ ! -f /tmp/tensorrt_edgellm/.done ]; then \
      echo "FAILED to install TensorRT-Edge-LLM ${TENSORRT_EDGELLM_VERSION}"; \
      exit 1; \
    fi

ENV TENSORRT_EDGELLM_DIR=${SOURCE_DIR} \
    PATH=${PATH}:${SOURCE_DIR}/build/examples/llm:${SOURCE_DIR}/build/examples/multimodal
