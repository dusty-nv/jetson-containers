#---
# name: tensorrt_llm
# group: llm
# config: config.py
# depends: [cudastack:standard, pytorch, transformers, cuda-python, nixl]
# test: [test.py]
# requires: '>=35'
# notes: The `tensorrt-llm:builder` container includes the C++ binaries under `/opt`
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TRT_LLM_VERSION \
    TRT_LLM_BRANCH \
    TRT_LLM_SOURCE \
    TRT_LLM_PATCH \
    CUDA_ARCHS \
    CUDA_VERSION \
    FORCE_BUILD="off" \
    BUILD_DIR="/opt/TensorRT-LLM/cpp/build" \
    SOURCE_DIR="/opt/TensorRT-LLM" \
    SOURCE_TAR="/tmp/TensorRT-LLM/source.tar.gz" \
    GIT_PATCHES="/tmp/TensorRT-LLM/patch.diff" \
    TMP_DIR="/tmp/TensorRT-LLM/"

WORKDIR ${SOURCE_DIR}

RUN add-apt-repository main && \
    add-apt-repository universe && \
    add-apt-repository restricted && \
    add-apt-repository multiverse && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    openmpi-bin \
    libopenmpi-dev \
    git-lfs \
    ccache && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/lib/ccache:$PATH"


RUN git clone --recurse-submodules --branch ${TRT_LLM_BRANCH} https://github.com/NVIDIA/TensorRT-LLM ${SOURCE_DIR} || \
    git clone --recurse-submodules https://github.com/NVIDIA/TensorRT-LLM.git ${SOURCE_DIR} ; \
    git lfs pull

COPY ${TRT_LLM_PATCH} ${GIT_PATCHES}
COPY sources ${TMP_DIR}
COPY build.sh install.sh /tmp/setup/

RUN /tmp/setup/install.sh || /tmp/setup/build.sh #|| echo "BUILD FAILED!"

RUN uv pip install --upgrade transformers && \
    uv pip install uvicorn fastapi

COPY llama.sh ${SOURCE_DIR}/

# check python version and save variable PYTHON_VERSION
RUN PYTHON_TAG=$(uv run python -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))") \
    && echo "PYTHON_TAG=${PYTHON_TAG}"
ARG PYTHON_VERSION=${PYTHON_TAG}

# name '_device_get_memory_info_fn' is not defined
COPY patches/profiler.py /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt_llm/profiler.py
COPY patches/convert_utils.py /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt_llm/models/convert_utils.py
