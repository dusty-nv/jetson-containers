#---
# name: stable_diffusion_cpp:builder
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
    STABLE_DIFFUSION_PYTHON_REPO \
    STABLE_DIFFUSION_PYTHON_BRANCH \
    STABLE_DIFFUSION_PYTHON_DIR="/opt/llama-cpp-python"

# the llama-cpp-python bindings contain llama-cpp as submodule - use that version for sanity
ADD https://api.github.com/repos/${STABLE_DIFFUSION_PYTHON_REPO}/git/refs/heads/${STABLE_DIFFUSION_PYTHON_BRANCH} /tmp/stable_diffusion_cpp_python_version.json

RUN set -ex \
    && git clone --branch=${STABLE_DIFFUSION_PYTHON_BRANCH} --depth=1 --recursive https://github.com/${STABLE_DIFFUSION_PYTHON_REPO} ${STABLE_DIFFUSION_PYTHON_DIR} \
    && ln -s "$STABLE_DIFFUSION_PYTHON_DIR/vendor/llama.cpp" "$STABLE_DIFFUSION_PYTHON_DIR/llama.cpp" \
    \
    # build C++ libraries \
    && mkdir -p "$STABLE_DIFFUSION_PYTHON_DIR/vendor/llama.cpp/build" \
    && cmake \
        -B "$STABLE_DIFFUSION_PYTHON_DIR/vendor/llama.cpp/build" \
        -S "$STABLE_DIFFUSION_PYTHON_DIR/vendor/llama.cpp" \
        -DLLAMA_CUBLAS=on \
        -DLLAMA_CUDA_F16=1 \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    && cmake \
        --build "$STABLE_DIFFUSION_PYTHON_DIR/vendor/llama.cpp/build" \
        --config Release \
        --parallel $(nproc) \
    && ln -s $STABLE_DIFFUSION_PYTHON_DIR/vendor/llama.cpp/build/bin $STABLE_DIFFUSION_PYTHON_DIR/llama.cpp/bin \
    \
    # build Python bindings \
    && CMAKE_ARGS="-DLLAMA_CUBLAS=on -DLLAMA_CUDA_F16=1 -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" FORCE_CMAKE=1 \
        uv build --wheel --out-dir /opt --verbose "$STABLE_DIFFUSION_PYTHON_DIR" \
    \
    # install the wheel \
    # python3 -m stable_diffusion_cpp.server missing 'import uvicorn' \
    && uv pip install \
        /opt/stable_diffusion_cpp_python*.whl \
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
    && python3 -c 'import stable_diffusion_cpp' \
    && python3 -m stable_diffusion_cpp.server --help
