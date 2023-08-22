#---
# name: llama_cpp
# group: llm
# config: config.py
# depends: [cmake, python, numpy, huggingface_hub]
# requires: '>=34.1.0'
# test:
#   - 'test.py --model $(huggingface-downloader TheBloke/Llama-2-7B-GGML/llama-2-7b.ggmlv3.q4_0.bin)'
#   - 'test_tokenizer.py --model $(huggingface-downloader TheBloke/Llama-2-7B-GGML/llama-2-7b.ggmlv3.q4_0.bin)'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_ARCHITECTURES

WORKDIR /opt

# clone it (force rebuild on new git commits)
#ADD https://api.github.com/repos/ggerganov/llama.cpp/git/refs/heads/master /tmp/llama_cpp_version.json
#RUN git clone --depth=1 --recursive https://github.com/ggerganov/llama.cpp

# the llama-cpp-python bindings contain llama-cpp as submodule - use that version for sanity
ADD https://api.github.com/repos/abetlen/llama-cpp-python/git/refs/heads/main /tmp/llama_cpp_python_version.json
RUN git clone --depth=1 --recursive https://github.com/abetlen/llama-cpp-python
RUN ln -s llama-cpp-python/vendor/llama.cpp llama.cpp

# apply patches
COPY patches.diff llama.cpp/

RUN cd llama.cpp && \
    git apply patches.diff && \
    git diff
    
# llama.cpp/ggml.h(260): error: identifier "__fp16" is undefined
# these are now applied in patch.diff (along with other patches)
#RUN cd llama.cpp && \
#    sed 's|#ifdef __ARM_NEON|#if defined(__ARM_NEON) \&\& !defined(__CUDACC__)|g' -i llama.cpp/ggml.h && \
#    cat llama.cpp/ggml.h | grep '__ARM_NEON'

# build C++ libraries
RUN cd llama-cpp-python/vendor/llama.cpp && \
    mkdir build && \
    cd build && \
    cmake .. -DLLAMA_CUBLAS=on -DLLAMA_CUDA_F16=1 -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    cmake --build . --config Release --parallel $(nproc)
    
RUN ln -s build/bin llama.cpp/bin

# apply patches
#RUN cd llama-cpp-python/vendor/llama.cpp && \
#    git apply /opt/llama.cpp/patches.diff && \
#    git diff
    
# build Python bindings
RUN cd llama-cpp-python && \
    CMAKE_ARGS="-DLLAMA_CUBLAS=on -DLLAMA_CUDA_F16=1 -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" FORCE_CMAKE=1 \
    pip3 wheel -w dist --verbose . 
 
# install the wheel
RUN cp llama-cpp-python/dist/llama_cpp_python*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/llama_cpp_python*.whl

# python3 -m llama_cpp.server missing 'import uvicorn'
RUN pip3 install --no-cache-dir --verbose uvicorn anyio starlette sse-starlette fastapi pydantic-settings

# for benchmark timing
RUN apt-get update && \
    apt-get install -y --no-install-recommends time \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
COPY benchmark.py llama.cpp/bin/benchmark.py

WORKDIR / 

# make sure it loads
RUN pip3 show llama-cpp-python | grep llama && \
    python3 -c 'import llama_cpp' && \
    python3 -m llama_cpp.server --help

