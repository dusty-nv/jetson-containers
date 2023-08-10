#---
# name: llama_cpp
# group: llm
# config: config.py
# depends: [cmake, python, numpy, huggingface_hub]
# requires: '>=34.1.0'
# test: test.py
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_ARCHITECTURES

WORKDIR /opt

# force rebuild on new git commits - https://stackoverflow.com/a/56945508
ADD https://api.github.com/repos/ggerganov/llama.cpp/git/refs/heads/master /tmp/llama_cpp_version.json

# llama.cpp/ggml.h(260): error: identifier "__fp16" is undefined
RUN git clone --depth=1 --recursive https://github.com/ggerganov/llama.cpp && \
    sed 's|#ifdef __ARM_NEON|#if defined(__ARM_NEON) \&\& !defined(__CUDACC__)|g' -i llama.cpp/ggml.h && \
    cat llama.cpp/ggml.h | grep '__ARM_NEON'

RUN cd llama.cpp && \
    mkdir build && \
    cd build && \
    cmake .. -DLLAMA_CUBLAS=on -DLLAMA_CUDA_F16=1 -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    cmake --build . --config Release --parallel $(nproc)
    
#RUN cmake --install llama.cpp/build
RUN ln -s /opt/llama.cpp/build/bin /opt/llama.cpp/bin

# Python bindings
ADD https://api.github.com/repos/abetlen/llama-cpp-python/git/refs/heads/main /tmp/llama_cpp_python_version.json
RUN git clone --depth=1 --recursive https://github.com/abetlen/llama-cpp-python

# llama.cpp/ggml.h(260): error: identifier "__fp16" is undefined
RUN sed 's|#ifdef __ARM_NEON|#if defined(__ARM_NEON) \&\& !defined(__CUDACC__)|g' -i llama-cpp-python/vendor/llama.cpp/ggml.h

RUN cd llama-cpp-python && \
    CMAKE_ARGS="-DLLAMA_CUBLAS=on -DLLAMA_CUDA_F16=1 -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" FORCE_CMAKE=1 \
    pip3 wheel -w dist --verbose . 
 
RUN cp llama-cpp-python/dist/llama_cpp_python*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/llama_cpp_python*.whl

# for benchmark timing
RUN apt-get update && \
    apt-get install -y --no-install-recommends time \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
COPY benchmark.py /opt/llama.cpp/bin/benchmark.py

WORKDIR / 

RUN pip3 show llama-cpp-python | grep llama && python3 -c 'import llama_cpp'

