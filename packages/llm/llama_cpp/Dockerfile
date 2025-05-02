#---
# name: llama_cpp
# group: llm
# config: config.py
# depends: [cuda, cudnn, cmake, python, numpy, huggingface_hub, sudonim]
# requires: '>=34.1.0'
# test: test_version.py
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG LLAMA_CPP_VERSION \
    LLAMA_CPP_VERSION_PY \
    LLAMA_CPP_BRANCH \
    LLAMA_CPP_BRANCH_PY \
    LLAMA_CPP_FLAGS \
    FORCE_BUILD=off \ 
    SOURCE_DIR=/opt/llama_cpp_python \
    TMP=/tmp/llama_cpp

COPY build.sh install.sh $TMP/
COPY benchmark.py /usr/local/bin/llama_cpp_benchmark.py

RUN $TMP/install.sh || $TMP/build.sh

RUN mkdir -p /root/.cache && \
    ln -s /data/models/llama.cpp /root/.cache/llama.cpp 