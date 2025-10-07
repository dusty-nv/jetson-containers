#---
# name: llama_cpp:builder
# group: llm
# config: config.py
# depends: [cuda, cudnn, cmake, python, numpy, huggingface_hub]
# requires: '>=34.1.0'
# test: test_version.py
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_ARCHITECTURES \
    LLAMA_CPP_PYTHON_REPO \
    LLAMA_CPP_PYTHON_BRANCH \
    LLAMA_CPP_PYTHON_DIR="/opt/llama-cpp-python"

# the llama-cpp-python bindings contain llama-cpp as submodule - use that version for sanity
ADD https://api.github.com/repos/${LLAMA_CPP_PYTHON_REPO}/git/refs/heads/${LLAMA_CPP_PYTHON_BRANCH} /tmp/llama_cpp_python_version.json

RUN set -ex \
    && git clone --branch=${LLAMA_CPP_PYTHON_BRANCH} --depth=1 --recursive https://github.com/${LLAMA_CPP_PYTHON_REPO} ${LLAMA_CPP_PYTHON_DIR} \
    && ln -s "$LLAMA_CPP_PYTHON_DIR/vendor/llama.cpp" "$LLAMA_CPP_PYTHON_DIR/llama.cpp" \
    \
    # build C++ libraries \
    && mkdir -p "$LLAMA_CPP_PYTHON_DIR/vendor/llama.cpp/build" \
    && cmake \
        -B "$LLAMA_CPP_PYTHON_DIR/vendor/llama.cpp/build" \
        -S "$LLAMA_CPP_PYTHON_DIR/vendor/llama.cpp" \
        -DLLAMA_CUBLAS=on \
        -DLLAMA_CUDA_F16=1 \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    && cmake \
        --build "$LLAMA_CPP_PYTHON_DIR/vendor/llama.cpp/build" \
        --config Release \
        --parallel $(nproc) \
    && ln -s $LLAMA_CPP_PYTHON_DIR/vendor/llama.cpp/build/bin $LLAMA_CPP_PYTHON_DIR/llama.cpp/bin \
    \
    # build Python bindings \
    && CMAKE_ARGS="-DLLAMA_CUBLAS=on -DLLAMA_CUDA_F16=1 -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" FORCE_CMAKE=1 \
        uv build --wheel --out-dir /opt --verbose "$LLAMA_CPP_PYTHON_DIR" \
    \
    # install the wheel \
    # python3 -m llama_cpp.server missing 'import uvicorn' \
    && uv pip install \
        /opt/llama_cpp_python*.whl \
        typing-extensions \
        uvicorn \
        anyio \
        starlette \
        sse-starlette \
        starlette-context \
        fastapi \
        pydantic-settings

# add benchmark script
COPY benchmark.py llama.cpp/bin/benchmark.py

# make sure it loads
RUN set -ex \
    && uv pip show llama-cpp-python | grep llama \
    && python3 -c 'import llama_cpp' \
    && python3 -m llama_cpp.server --help
