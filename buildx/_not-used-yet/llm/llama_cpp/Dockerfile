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
    LLAMA_CPP_BRANCH \
    LLAMA_CPP_FLAGS \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/llama_cpp/
COPY benchmark.py /usr/local/bin/llama_cpp_benchmark.py

RUN /tmp/llama_cpp/install.sh || /tmp/llama_cpp/build.sh
RUN mkdir -p /root/.cache && \
    ln -s /data/models/llama.cpp /root/.cache/llama.cpp 