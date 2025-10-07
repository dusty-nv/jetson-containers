#---
# name: nvidia_modelopt
# group: llm
# config: config.py
# depends: [cudastack:standard, pytorch, torchvision, torchaudio, transformers, cuda-python, tensorrt_llm]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/NVIDIA/TensorRT-Model-Optimizer
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NVIDIA_MODELOPT_VERSION \
    CUDAARCHS

COPY build.sh install.sh /tmp/NVIDIA_MODELOPT/

COPY benchmarks /tmp/NVIDIA_MODELOPT/benchmarks

RUN /tmp/NVIDIA_MODELOPT/install.sh || /tmp/NVIDIA_MODELOPT/build.sh
