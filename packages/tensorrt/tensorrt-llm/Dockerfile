#---
# name: tensorrt-llm
# group: llm
# config: config.py
# depends: [tensorrt, pytorch, transformers, cuda-python, cmake]
# test: test.py
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TRT_LLM_VERSION
ARG TRT_LLM_PATCH

ARG CUDA_ARCHITECTURES

WORKDIR /opt

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  git-lfs \
		  mpich \
		  libmpich-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
RUN git clone --branch v${TRT_LLM_VERSION} --depth 1 --recursive https://github.com/NVIDIA/TensorRT-LLM tensorrt-llm && \
    cd tensorrt-llm && \
    git lfs install && \
    git lfs pull 
    
COPY ${TRT_LLM_PATCH} tensorrt-llm/patch.diff

RUN cd tensorrt-llm && \
    git apply patch.diff

RUN cd tensorrt-llm && \
    python3 scripts/build_wheel.py \
        --cuda_architectures ${CUDA_ARCHITECTURES} \
	   -D ENABLE_MULTI_DEVICE=0

RUN cp tensorrt-llm/build/*.whl . && \
    pip3 install --no-cache-dir --verbose tensorrt_llm*.whl
    
RUN pip3 show tensorrt_llm && python3 -c 'import tensorrt_llm'

RUN pip3 install --no-cache-dir pydantic pynvml

WORKDIR /
