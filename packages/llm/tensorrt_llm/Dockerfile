#---
# name: tensorrt_llm
# group: llm
# config: config.py
# depends: [tensorrt, pytorch, transformers, cuda-python]
# test: [test.py, test.sh]
# requires: '>=35'
# notes: The `tensorrt-llm:builder` container includes the C++ binaries under `/opt`
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TRT_LLM_VERSION \
    TRT_LLM_BRANCH \
    TRT_LLM_PATCH \
    CUDA_ARCHS \
    FORCE_BUILD="off" \
    SRC_DIR="/opt/tensorrt_llm" \
    BUILD_DIR="/opt/tensorrt_llm/cpp/build"
 
#COPY build.sh install.sh /tmp/tensorrt_llm/ 
#COPY ${TRT_LLM_PATCH} /tmp/tensorrt_llm/patch.diff

#RUN /tmp/tensorrt_llm/install.sh || /tmp/tensorrt_llm/build.sh

# this is just temporary to aid in faster builds/patching
COPY install.sh /tmp/tensorrt_llm/ 
RUN /tmp/tensorrt_llm/install.sh || echo "pip install failed, building from source"

COPY build.sh /tmp/tensorrt_llm/ 
COPY ${TRT_LLM_PATCH} /tmp/tensorrt_llm/patch.diff

RUN /tmp/tensorrt_llm/build.sh