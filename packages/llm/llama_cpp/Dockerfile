#---
# name: llama_cpp
# group: llm
# config: config.py
# depends: [cmake, python, numpy, huggingface_hub]
# requires: '>=34.1.0'
# test: test_version.py
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_ARCHITECTURES

ARG LLAMA_CPP_PYTHON_REPO
ARG LLAMA_CPP_PYTHON_BRANCH

WORKDIR /opt

# the llama-cpp-python bindings contain llama-cpp as submodule - use that version for sanity
ADD https://api.github.com/repos/${LLAMA_CPP_PYTHON_REPO}/git/refs/heads/${LLAMA_CPP_PYTHON_BRANCH} /tmp/llama_cpp_python_version.json
RUN git clone --branch=${LLAMA_CPP_PYTHON_BRANCH} --depth=1 --recursive https://github.com/${LLAMA_CPP_PYTHON_REPO}
RUN ln -s llama-cpp-python/vendor/llama.cpp llama.cpp

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
RUN pip3 install --no-cache-dir --verbose uvicorn anyio starlette sse-starlette starlette-context fastapi pydantic-settings

# add benchmark script
COPY benchmark.py llama.cpp/bin/benchmark.py

WORKDIR / 

# make sure it loads
RUN pip3 show llama-cpp-python | grep llama && \
    python3 -c 'import llama_cpp' && \
    python3 -m llama_cpp.server --help

