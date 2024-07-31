#---
# name: llama_cpp
# group: llm
# config: config.py
# depends: [cuda, cudnn, cmake, python, numpy, huggingface_hub]
# requires: '>=34.1.0'
# test: test_version.py
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG LLAMA_CPP_VERSION \
    LLAMA_CPP_BRANCH \
    LLAMA_CPP_FLAGS

COPY build.sh /tmp/build_llama_cpp.sh
COPY benchmark.py /usr/local/bin/llama_cpp_benchmark.py

RUN pip3 install --no-cache-dir --verbose \
        typing-extensions \
        uvicorn \
        anyio \
        starlette \
        sse-starlette \
        starlette-context \
        fastapi \
        pydantic-settings && \
    pip3 install --no-cache-dir --verbose llama-cpp-python==${LLAMA_CPP_VERSION} || \
    /tmp/build_llama_cpp.sh
    
